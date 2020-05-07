#!/usr/bin/env python3
"""
MRAC prediction using Deep Learning 3D U-net
Author: Claes Ladefoged, Rigshospitalet, Copenhagen, Denmark
        claes.noehr.ladefoged@regionh.dk
Version: August-20-2019

Input: 
        path : Path to folder with dicom files of Dixon opposed-phase, in-phase, and Umap
        model : Path to model, or 'ALL' if all 4 DeepDixon models is to be used. [Default: model1].
        output: Path to output folder [Default: DeepDixon]
Output: Folder with dicom files of MRAC DeepDixon within input folder path
"""

import argparse, os, shutil, datetime
import numpy as np 
import pydicom as dicom  
import nibabel as nib
import dicom2nifti
from rhscripts.DeepMRAC import predict_DeepT1
import tempfile
from nilearn.image import resample_img

""" Settings """
tmpdir = ''
verbose = False

"""
Sort the files in the input folder based on instance number.
Store the files in a temporary folder
"""
def sort_files(folder):
    os.mkdir(f'{tmpdir}/dicom')
    
    for root,subdirs,files in os.walk(folder):
        
        if len(subdirs) > 0:
            continue
        if not len(files) > 0:
            continue
        
        if verbose:
            print("Found files in %s. Making copy" % root)
        
        for f in files:
            if f.startswith('.'):
                continue
            dcm = dicom.read_file("%s/%s" % (root,f))

            shutil.copyfile(os.path.join(root,f),"%s/dicom/dicom%000d.ima" % (tmpdir,int(dcm.InstanceNumber)))
    
"""
Check that the correct number of files is present, and convert the images to nifti
"""
def convert_data():
    if verbose:
        print('Converting to nifti')
    
    dicom2nifti.dicom_series_to_nifti(f'{tmpdir}/dicom', f'{tmpdir}/t1.nii.gz', reorient_nifti=True)

"""
Resample the images to have isotropic voxel size (1.5626) and 192^3 matrix size
"""
def load_and_resample_images():
    
    if verbose:
        print("Loading and resampling data")
        
    # Load dataset
    t1 = nib.load(f'{tmpdir}/t1.nii.gz')

    # Resample to 192x192x192 and isotropic voxel size of 1.5626    
    target_shape = np.array((192,192,192))
    new_resolution = [1.5626,-1.5626,-1.5626]
    new_affine = np.zeros((4,4))
    new_affine[:3,:3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3,3] = target_shape*new_resolution/2.*-1
    new_affine[3,3] = 1.
    t1_rsl = resample_img(t1, target_affine=new_affine, target_shape=target_shape, interpolation='linear')
    
    # Flip to match orientation on what was trained on
    img = np.flipud(np.swapaxes(t1_rsl.get_fdata(),0,2))
    
    # Return nii handles as well as new image
    return t1, t1_rsl, img

"""
Resample the predicted image back to dixon resolution and resample to umap format.
"""
def resample_to_native(DeepX, reference, reference_native, save_prediction=False):
    if verbose:
        print('Resampling back to native resolution')
    
    # Rehape to native resolution
    DeepX = np.swapaxes(np.flipud(DeepX),2,0)
    DeepX_nii = nib.Nifti1Image(DeepX, reference.affine, reference.header)
    DeepX_rsl = resample_img(DeepX_nii,target_affine=reference_native.affine,target_shape=reference_native.shape,interpolation='linear')
    
    # Save intermediate nii file
    if save_prediction:
        if verbose:
            print("Saving QC nii file to DeepT1_QC.nii.gz")
        nib.save(DeepX_rsl,'DeepT1_QC.nii.gz')
    
    return DeepX_rsl.get_fdata()

"""
Overwrite the container dicom files with
    PixelData from predicted numpy array
    SeriesInstanceUID and SOPInstanceUID to make it unique
    Description and Number
Saves new dicom series    
"""
def to_dcm(DeepX,dcmcontainer,dicomfolder):  
    
    def listdir_nohidden(path):
        return [f for f in os.listdir(path) if not f.startswith('.')]
    
    # Read first file to get header information
    ds=dicom.read_file(os.path.join(dcmcontainer,'dicom1.ima'))
    LargestImagePixelValue = DeepX.max()
    
    # Fix orientation due to nifti ( perhaps this could be done smarter.. )
    DeepX = np.fliplr(np.flipud(np.swapaxes(DeepX,2,0)))
    np_DeepX = np.array(DeepX,dtype=ds.pixel_array.dtype)

    # Generate unique SeriesInstanceUID
    newSIUID = str(datetime.datetime.now())
    newSIUID = newSIUID.replace("-","")
    newSIUID = newSIUID.replace(" ","")
    newSIUID = newSIUID.replace(":","")
    newSIUID = newSIUID.replace(".","")
    newSIUID = '1.3.12.2.1107.5.2.38.51014.' + str(newSIUID) + '11111.0.0.0' 

    if not os.path.exists(dicomfolder):
        os.mkdir(dicomfolder)

    # Read each file in UMAP container, replace relevant tags
    for f in listdir_nohidden(dcmcontainer):
        ds=dicom.read_file(os.path.join(dcmcontainer,f))
        i = int(ds.InstanceNumber)-1
        
        ds.LargestImagePixelValue = int(LargestImagePixelValue)
        ds.PixelData = np_DeepX[:,:,i].tostring() # Inserts actual image info

        ds.SeriesInstanceUID = newSIUID
        ds.SeriesDescription = "DeepT1"
        ds.SeriesNumber = "507"

        # Generate unique SOPInstanceUID
        newSOP = str(datetime.datetime.now())
        newSOP = newSOP.replace("-","")
        newSOP = newSOP.replace(" ","")
        newSOP = newSOP.replace(":","")
        newSOP = newSOP.replace(".","")
        newSOP = '1.3.12.2.1107.5.2.38.51014.' + str(newSOP) + str(i+1)
        ds.SOPInstanceUID = newSOP

        # Save the file
        fname = "dicom_%04d.dcm" % int(ds.InstanceNumber)
        ds.save_as(os.path.join(dicomfolder,fname))
    
""" 
Function to keep track of where to store the data
Actual storing is performed in to_dcm()
"""
def convert_to_DCM(DeepX,patient,folder_outname):
    if verbose:
        print('Saving data to dicom')
    
    # Determine where to store the output dicom files
    if folder_outname == None:
       outname = os.path.join(patient,'DeepT1')
    else:
       outname = folder_outname
    
    # Convert the files
    to_dcm(DeepX,os.path.join(tmpdir,'dicom'),outname)

""" 
Function to check output
"""

def check_output(x):
    try:
        x.shape
        return True
    except:
        return False
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict using DeepDixon.')
    parser.add_argument("patient", help="Path to patient.")
    parser.add_argument("--outname", help="Name for output folder. ", type=str)
    parser.add_argument("--version", help="Software version used to train the model (VB20P or VE11P) Default: VE11P. ", type=str, default='VE11P')
    parser.add_argument("--save_nii", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    # Set verbose level
    verbose = args.verbose
    
    # Create temporary folder    
    tmpdir = tempfile.mkdtemp()
    
    # Sort files into specific folders
    sort_files(args.patient)
    
    # Load and resample data
    convert_data()
    t1, t1_rsl, img = load_and_resample_images()
    
    # Predict
    if verbose:
        print("Predicting DeepT1 using %s model" % args.version)
    DeepX = predict_DeepT1(img,args.version)
    if not check_output(DeepX):
        shutil.rmtree(tmpdir)
        exit(-1)
    
    # Reshape image to native resolution
    DeepX = resample_to_native(DeepX, t1_rsl, t1, args.save_nii)
    
    # Convert to DICOM
    convert_to_DCM(DeepX,args.patient,args.outname)
    
    # Cleanup
    shutil.rmtree(tmpdir)
