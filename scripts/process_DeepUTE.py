#!/usr/bin/env python3
"""
MRAC prediction using Deep Learning 3D U-net
Author: Claes Ladefoged, Rigshospitalet, Copenhagen, Denmark
        claes.noehr.ladefoged@regionh.dk
Version: March-12-2019

Input: 
        path : Path to folder with dicom files of UTE Echo 1, Echo 2 and Umap
        model : Path to model, or 'ALL' if all 4 DeepUTE models is to be used. [Default: model1].
        output: Path to output folder [Default: DeepUTE]
Output: Folder with dicom files of MRAC DeepUTE within input folder path
"""

import argparse, os, glob, shutil, datetime, subprocess
import numpy as np
from numpy.lib import stride_tricks 
from keras.models import model_from_json 
import pydicom as dicom  

""" Settings """
# Path to current folder
root_folder = '/users/claes/projects/MRAC_routine/pipeline'
# Settings for patch extraction
h = 16 # Number of slices 
sh = 2 # Patch stride
    
"""
Sort the files in the input folder based on series and instance number.
Store the files in a temporary folder
"""
def sort_files(folder):
    
    if os.path.exists('%s/tmp' % root_folder):
        shutil.rmtree('%s/tmp' % root_folder)
    os.mkdir("%s/tmp" % root_folder)
    
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
            
            if not os.path.exists("%s/tmp/%s" % (root_folder,dcm.SeriesNumber)):
                os.mkdir("%s/tmp/%s" % (root_folder,dcm.SeriesNumber))
            
            shutil.copyfile(os.path.join(root,f),"%s/tmp/%s/dicom%000d.ima" % (root_folder,dcm.SeriesNumber,int(dcm.InstanceNumber)))
    
"""
Check that the correct number of files is present, and load the UTE TE1 and TE2 images
"""
def load_data():
    utes = [ f for f in os.listdir('%s/tmp' % root_folder) if not f.startswith('.') ]
    utes.sort(key=float)
    
    # Check that correct number of files is present
    assert len(os.listdir('%s/tmp/%s' % (root_folder, utes[0]))) == 192
    assert len(os.listdir('%s/tmp/%s' % (root_folder, utes[1]))) == 192
    assert len(os.listdir('%s/tmp/%s' % (root_folder, utes[2]))) == 192
    
    # Load UTE TE1
    ute1 = np.empty((192,192,192))
    for filename in os.listdir('%s/tmp/%s' % (root_folder, utes[0])):
        ds = dicom.dcmread('%s/tmp/%s/%s' % (root_folder, utes[0], filename))
        i = int(ds.InstanceNumber)-1
        ute1[i,:,:] = ds.pixel_array
        
    # Load UTE TE2
    ute2 = np.empty((192,192,192))
    for filename in os.listdir('%s/tmp/%s' % (root_folder, utes[1])):
        ds = dicom.dcmread('%s/tmp/%s/%s' % (root_folder, utes[1], filename))
        i = int(ds.InstanceNumber)-1
        ute2[i,:,:] = ds.pixel_array
        
    return ute1,ute2

"""
Return all combinations of 192x192x16 patches from each TE image with stride 2
"""
def get_patches_znorm():
    
    # Load the UTE TE images
    ute1, ute2 = load_data()
    
    # Standardize with z-norm using UTE TE2 meand and std only
    mean_UTE2, std_UTE2 = ( np.mean(ute2[np.where(ute2>0)]), np.std(ute2[np.where(ute2>0)]) )
    ute1 = np.true_divide( ute1 - mean_UTE2, std_UTE2 )
    ute2 = np.true_divide( ute2 - mean_UTE2, std_UTE2 )

    # Extract patches
    patches_UTE1 = cutup(ute1,(16,192,192),(2,1,1))
    patches_UTE2 = cutup(ute2,(16,192,192),(2,1,1))
    
    # Combine images into matrix of TE1 and TE2 patches
    ijk = patches_UTE1.shape[0]*patches_UTE1.shape[1]*patches_UTE1.shape[2] # Number of patches
    selected_patches = np.empty((ijk,16,192,192,2), dtype='float32')
    selected_patches[:,:,:,:,0] = np.reshape(patches_UTE1,(ijk,16,192,192))
    selected_patches[:,:,:,:,1] = np.reshape(patches_UTE2,(ijk,16,192,192))
    selected_patches = selected_patches.astype('float32')

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
def _predict(model):
    
    # Container matrices for data and counter for overlapping patches
    predicted_combined = np.zeros((192,192,192))
    predicted_counter = np.zeros((192,192,192))

    # Load all patches
    selected_patches = get_patches_znorm()

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
def predict_DeepUTE(model):

    if model == None:
        KERAS_models = ['models/DeepUTE/DeepUTE_VE11P_model1'] # CHANGED TO THIS VERSION 06-03-2019]
    elif model == "ALL":
        KERAS_models = [ os.path.splitext(m)[0] for m in glob.glob('models/DeepUTE/*.h5') ]
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
        predicted_output += _predict(model)

    # Return averaged output			
    return predicted_output / float(len(KERAS_models))
    
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
    
    utes = [ f for f in os.listdir('%s/tmp' % root_folder) if not f.endswith('.mnc') or f.startswith('.') ]
    utes.sort(key=float)
    
    # Determine where to store the output dicom files
    if folder_outname == None:
       outname = os.path.join(patient,'DeepUTE')
    else:
       outname = folder_outname
    
    # Convert the files
    to_dcm(DeepX,os.path.join(root_folder,'tmp',utes[2]),outname)
    
    # Clean up the tmp files
    if os.path.exists('%s/tmp' % root_folder):
        shutil.rmtree('%s/tmp' % root_folder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train RF model or predict using existing model.')
    parser.add_argument("patient", help="Path to patient.")
    parser.add_argument("--model", help="Use the KERAS model weights as initializer. Default: ", type=str)
    parser.add_argument("--outname", help="Name for output folder. ", type=str)
    args = parser.parse_args()
    
    # Sort files into specific folders
    sort_files(args.patient)
    
    # Predict
    DeepX = predict_DeepUTE(args.model)
    
    # Convert to DICOM
    convert_to_DCM(DeepX,args.patient,args.outname)
    
    
