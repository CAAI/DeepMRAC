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

import argparse, os, glob, shutil, datetime
import numpy as np
from numpy.lib import stride_tricks 
from keras.models import model_from_json 
import pydicom as dicom  
import nibabel as nib
import dicom2nifti
from nipype.interfaces.fsl import FLIRT

""" Settings """
# Path to current folder
root_folder = '/users/claes/projects/MRAC_routine/pipeline/tmp'
# Settings for patch extraction
h = 16 # Number of slices 
sh = 2 # Patch stride
verbose = False
    
"""
Sort the files in the input folder based on series and instance number.
Store the files in a temporary folder
"""
def sort_files(folder):
    
    if os.path.exists('%s' % root_folder):
        shutil.rmtree('%s' % root_folder)
    os.mkdir("%s" % root_folder)
    
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
            
            if not os.path.exists("%s/%s" % (root_folder,dcm.SeriesNumber)):
                os.mkdir("%s/%s" % (root_folder,dcm.SeriesNumber))
            
            shutil.copyfile(os.path.join(root,f),"%s/%s/dicom%000d.ima" % (root_folder,dcm.SeriesNumber,int(dcm.InstanceNumber)))
    
"""
Check that the correct number of files is present, and convert the images to nifti
"""
def convert_data():
    if verbose:
        print('Sorting the dicom files and converting to nifti')
    datasets = [ f for f in os.listdir(root_folder) if not f.startswith('.') ]
    datasets.sort(key=float)
    
    # Check that correct number of files is present
    assert len(os.listdir('%s/%s' % (root_folder, datasets[0]))) == 128
    assert len(os.listdir('%s/%s' % (root_folder, datasets[1]))) == 128
    assert len(os.listdir('%s/%s' % (root_folder, datasets[2]))) == 132
    
    
    dicom2nifti.dicom_series_to_nifti('%s/%s' % (root_folder, datasets[0]), '%s/opposedphase.nii.gz' % root_folder, reorient_nifti=True)
    dicom2nifti.dicom_series_to_nifti('%s/%s' % (root_folder, datasets[1]), '%s/inphase.nii.gz' % root_folder, reorient_nifti=True)
    dicom2nifti.dicom_series_to_nifti('%s/%s' % (root_folder, datasets[2]), '%s/umap.nii.gz' % root_folder, reorient_nifti=True)

"""
Resample the images to have isotropic voxel size (1.3021) - only z will change
"""
def isotropic_voxels():
    if verbose:
        print('Resampling to isotropic voxels')
    orig = nib.load('%s/opposedphase.nii.gz' % root_folder)
    
    for f in ['opposedphase','inphase']:        
        iso = FLIRT()
        iso.inputs.in_file = '%s/%s.nii.gz' % (root_folder,f)
        iso.inputs.reference = '%s/%s.nii.gz' % (root_folder,f)
        iso.inputs.out_file = '%s/%s_iso.nii.gz' % (root_folder,f)
        iso.inputs.apply_isoxfm = orig.header.get_zooms()[0]
        iso.run()

"""
Load the two nifti datasets
"""
def load_data():
    # Load Opposed-phase
    dataset1 = nib.load('%s/opposedphase_iso.nii.gz' % root_folder)

    # Load IN-phase
    dataset2 = nib.load('%s/inphase_iso.nii.gz' % root_folder)
        
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
Return all combinations of 192x192x16 patches from each TE image with stride 2
"""
def get_patches_znorm(vol1,vol2):
    mean_vol1, std_vol1 = ( np.mean(vol1[np.where(vol1>0)]), np.std(vol1[np.where(vol1>0)]) )
    mean_vol2, std_vol2 = ( np.mean(vol2[np.where(vol2>0)]), np.std(vol2[np.where(vol2>0)]) )
    
    # STANDARDIZE
    vol1 = np.true_divide( vol1 - mean_vol1, std_vol1 )
    vol2 = np.true_divide( vol2 - mean_vol2, std_vol2 )

    patches_vol1 = cutup(vol1,(16,192,192),(2,1,1))
    patches_vol2 = cutup(vol2,(16,192,192),(2,1,1))

    ijk = patches_vol1.shape[0]*patches_vol1.shape[1]*patches_vol1.shape[2]
    selected_patches = np.empty((ijk,16,192,192,2), dtype='float32')
    selected_patches[:,:,:,:,0] = np.reshape(patches_vol1,(ijk,16,192,192))
    selected_patches[:,:,:,:,1] = np.reshape(patches_vol2,(ijk,16,192,192))

    return selected_patches

"""
Helper function to cut patches out of data
"""
def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6
    
"""
Predicts patches of pCT data from patches of UTE data, 
and combines the result by averaging overlapping patches
"""
def _predict(model, vol1, vol2):
    
    # Container matrices for data and counter for overlapping patches
    predicted_combined = np.zeros((192,192,192))
    predicted_counter = np.zeros((192,192,192))

    # Load all patches
    selected_patches = get_patches_znorm(vol1, vol2)

    # Process a patch at a time
    for p in range(selected_patches.shape[0]):
        from_h = p*sh # Start slice of patch
        predicted = model.predict(np.reshape(selected_patches[p,:,:,:,:],(1,16,192,192,2))) # Predict pCT for patch
        predicted[ predicted == np.nan ] = -1 # Can occur, remove so output does not fail, but set to a value that can be searched for
        predicted_combined[from_h:from_h+h,:,:] += np.reshape(predicted,(16,192,192)) # Insert into container
        predicted_counter[from_h:from_h+h,:,:] += 1 # Update counter in area of patch for later average

    predicted_combined = np.divide(predicted_combined,predicted_counter) # Average over overlapping patches
    predicted_combined[ predicted_combined == np.inf ] = 0 # If divide by zero, remove here (should not occur since counter >> 0).
    
    return predicted_combined
    
"""
Predicts the DeepUTE pCT image with the model(s) specified
    If none is given, the first of 4 models is used
    If 'ALL' is given, all 4 models is used in an ensamble manor, and the result is averaged
    Otherwise, the path to another model can be specified
"""
def predict_DeepDixon(model, vol1, vol2):
    if verbose:
        print('Predicting attenuation map')
    if model == None:
        KERAS_models = ['models/DeepDixon/DeepDixon_VE11P_model1'] # CHANGED TO THIS VERSION 06-03-2019]
    elif model == "ALL":
        KERAS_models = [ os.path.splitext(m)[0] for m in glob.glob('models/DeepDixon/*.h5') ]
    else:
        KERAS_models = [ os.path.splitext(model)[0] ]
      
    # Container for output
    predicted_output = np.zeros((192,192,192))
    
    # Predict output for each of the given models (default is only 1)
    for KERAS_model in KERAS_models:
        # load json and create model
        json_file = open(KERAS_model+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(KERAS_model+".h5")
        predicted_output += _predict(model, vol1, vol2)

    # Return averaged output			
    return predicted_output / float(len(KERAS_models))

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
    nib.save(DeepX_nii,'%s/DeepDixon.nii.gz' % root_folder)
    
    # Resample to UMAP resolution
    DeepX_rsl = FLIRT()
    DeepX_rsl.inputs.in_file = '%s/DeepDixon.nii.gz' % root_folder
    DeepX_rsl.inputs.reference = '%s/umap.nii.gz' % root_folder
    DeepX_rsl.inputs.out_file = '%s/DeepDixon_rsl.nii.gz' % root_folder
    DeepX_rsl.run()
    DeepX = nib.load('%s/DeepDixon_rsl.nii.gz' % root_folder).get_fdata()
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
        ds.SeriesDescription = "DeepUTE"
        ds.SeriesNumber = "505"

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
    datasets = [ f for f in os.listdir(root_folder) if not f.endswith('.nii.gz') or f.startswith('.') ]
    datasets.sort(key=float)
    
    # Determine where to store the output dicom files
    if folder_outname == None:
       outname = os.path.join(patient,'DeepDixon')
    else:
       outname = folder_outname
    
    # Convert the files
    to_dcm(DeepX,os.path.join(root_folder,datasets[2]),outname)
    
    # Clean up the tmp files
    if os.path.exists('%s' % root_folder):
        shutil.rmtree('%s' % root_folder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train RF model or predict using existing model.')
    parser.add_argument("patient", help="Path to patient.")
    parser.add_argument("--model", help="Use the KERAS model weights as initializer. Default: ", type=str)
    parser.add_argument("--outname", help="Name for output folder. ", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    if args.verbose:
        verbose = True
    
    # Sort files into specific folders
    sort_files(args.patient)
    
    # Load and resample data
    convert_data()
    isotropic_voxels()
    opp_phase,in_phase = load_data()
    
    # Reshape images to 192**3
    opp_phase_rsl,in_phase_rsl = resample_images(opp_phase.get_fdata(),in_phase.get_fdata())
    
    # Predict
    DeepX = predict_DeepDixon(args.model, in_phase_rsl, opp_phase_rsl)
    
    # Reshape image to umap resolution
    DeepX = resample_to_umap(DeepX, in_phase)
    
    # Convert to DICOM
    convert_to_DCM(DeepX,args.patient,args.outname)
    
    
