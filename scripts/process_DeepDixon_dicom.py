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
from nipype.interfaces.fsl import FLIRT
from DeepMRAC import predict_DeepDixon
import tempfile

""" Settings """
tmpdir = ''
verbose = False
    
"""
Sort the files in the input folder based on series and instance number.
Store the files in a temporary folder
"""
def sort_files(folder):
    
    for root,subdirs,files in os.walk(folder):
        
        if len(subdirs) > 0:
            continue
        if not len(files) > 0:
            continue
        
        print("Found files in %s" % root)
        
        for f in files:
            if f.startswith('.'):
                continue
            dcm = dicom.read_file("%s/%s" % (root,f))
            
            if not os.path.exists("%s/%s" % (tmpdir,dcm.SeriesNumber)):
                os.mkdir("%s/%s" % (tmpdir,dcm.SeriesNumber))
            
            shutil.copyfile(os.path.join(root,f),"%s/%s/dicom%000d.ima" % (tmpdir,dcm.SeriesNumber,int(dcm.InstanceNumber)))
    
"""
Check that the correct number of files is present, and convert the images to nifti
"""
def convert_data():
    if verbose:
        print('Sorting the dicom files and converting to nifti')
    datasets = [ f for f in os.listdir(tmpdir) if not f.startswith('.') ]
    datasets.sort(key=float)
    
    # Check that correct number of files is present
    assert len(os.listdir('%s/%s' % (tmpdir, datasets[0]))) == 128
    assert len(os.listdir('%s/%s' % (tmpdir, datasets[1]))) == 128
    assert len(os.listdir('%s/%s' % (tmpdir, datasets[2]))) == 132
    
    
    dicom2nifti.dicom_series_to_nifti('%s/%s' % (tmpdir, datasets[0]), '%s/opposedphase.nii.gz' % tmpdir, reorient_nifti=True)
    dicom2nifti.dicom_series_to_nifti('%s/%s' % (tmpdir, datasets[1]), '%s/inphase.nii.gz' % tmpdir, reorient_nifti=True)
    dicom2nifti.dicom_series_to_nifti('%s/%s' % (tmpdir, datasets[2]), '%s/umap.nii.gz' % tmpdir, reorient_nifti=True)

"""
Resample the images to have isotropic voxel size (1.3021) - only z will change
"""
def isotropic_voxels():
    if verbose:
        print('Resampling to isotropic voxels')
    orig = nib.load('%s/opposedphase.nii.gz' % tmpdir)
    
    for f in ['opposedphase','inphase']:        
        iso = FLIRT()
        iso.inputs.in_file = '%s/%s.nii.gz' % (tmpdir,f)
        iso.inputs.reference = '%s/%s.nii.gz' % (tmpdir,f)
        iso.inputs.out_file = '%s/%s_iso.nii.gz' % (tmpdir,f)
        iso.inputs.apply_isoxfm = orig.header.get_zooms()[0]
        iso.run()

"""
Load the two nifti datasets
"""
def load_data():
    # Load Opposed-phase
    dataset1 = nib.load('%s/opposedphase_iso.nii.gz' % tmpdir)

    # Load IN-phase
    dataset2 = nib.load('%s/inphase_iso.nii.gz' % tmpdir)
        
    return dataset1, dataset2

"""
Reshape the images to match 192x192x192 matrix. 

OBS!! this cuts away part of the image!! For usual brain applications, this is only air.
Alternatively, use a sliding window predicting 192**3 blocks that can be averaged.
"""
def resample_images(dataset1,dataset2):
    if verbose:
        print('Reshaping to proper matrix size')
    sub1 = dataset1[96:288,6:198,:192]
    sub2 = dataset2[96:288,6:198,:192]
    
    swap1 = np.swapaxes(sub1,0,2)
    swap2 = np.swapaxes(sub2,0,2)
    
    flip1 = np.flipud(swap1)
    flip2 = np.flipud(swap2)
    
    flip1 = np.fliplr(flip1)
    flip2 = np.fliplr(flip2)
    
    return flip1,flip2

"""
Resample the predicted image back to dixon resolution and resample to umap format.
"""
def resample_to_umap(DeepX,reference):
    if verbose:
        print('Resampling to umap resolution')
    # Rehape to in-phase resolution
    DeepX = np.swapaxes(np.flipud(np.fliplr(DeepX)),2,0)
    DeepX_padded = np.zeros(reference.shape)
    DeepX_padded[96:288,6:198,:192] = DeepX
    DeepX_nii = nib.Nifti1Image(DeepX_padded,reference.affine, reference.header)
    nib.save(DeepX_nii,'%s/DeepDixon.nii.gz' % tmpdir)
    
    # Resample to UMAP resolution
    DeepX_rsl = FLIRT()
    DeepX_rsl.inputs.in_file = '%s/DeepDixon.nii.gz' % tmpdir
    DeepX_rsl.inputs.reference = '%s/umap.nii.gz' % tmpdir
    DeepX_rsl.inputs.out_file = '%s/DeepDixon_rsl.nii.gz' % tmpdir
    DeepX_rsl.run()
    DeepX = nib.load('%s/DeepDixon_rsl.nii.gz' % tmpdir).get_fdata()
    DeepX = DeepX.astype(np.uint16)
    DeepX = np.fliplr(np.flipud(np.swapaxes(np.swapaxes(DeepX,0,1),1,2)))
    
    return DeepX

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
        ds.PixelData = np_DeepX[i,:,:].tostring() # Inserts actual image info

        ds.SeriesInstanceUID = newSIUID
        ds.SeriesDescription = "DeepDixon"
        ds.SeriesNumber = "506"

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
    datasets = [ f for f in os.listdir(tmpdir) if not f.endswith('.nii.gz') or f.startswith('.') ]
    datasets.sort(key=float)
    
    # Determine where to store the output dicom files
    if folder_outname == None:
       outname = os.path.join(patient,'DeepDixon')
    else:
       outname = folder_outname
    
    # Convert the files
    to_dcm(DeepX,os.path.join(tmpdir,datasets[2]),outname)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict using DeepDixon.')
    parser.add_argument("patient", help="Path to patient.")
    parser.add_argument("--outname", help="Name for output folder. ", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    # Create temporary folder    
    tmpdir = tempfile.mkdtemp()
    
    # Sort files into specific folders
    sort_files(args.patient)
    
    # Load and resample data
    convert_data()
    isotropic_voxels()
    opp_phase,in_phase = load_data()
    
    # Reshape images to 192**3
    opp_phase_rsl,in_phase_rsl = resample_images(opp_phase.get_fdata(),in_phase.get_fdata())
    
    # Predict
    DeepX = predict_DeepDixon(in_phase_rsl, opp_phase_rsl)
    
    # Reshape image to umap resolution
    DeepX = resample_to_umap(DeepX, in_phase)
    
    # Convert to DICOM
    convert_to_DCM(DeepX,args.patient,args.outname)
    
    # Cleanup
    shutil.rmtree(tmpdir)
