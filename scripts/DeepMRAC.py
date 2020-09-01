# -*- coding: utf-8 -*-
"""
Created on Wed May  6 07:51:10 2020

@author: CLAD0003
"""

import numpy as np
import tensorflow as tf
from numpy.lib import stride_tricks 

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
Function to check output
"""

def check_output(x):
    try:
        x.shape
        return True
    except:
        return False

"""
Return all combinations of 192x192x16 patches from each image with stride 2
"""
def get_patches_znorm(vol1,vol2=None,normalize_both=True):
    mean_vol1, std_vol1 = ( np.mean(vol1[np.where(vol1>0)]), np.std(vol1[np.where(vol1>0)]) )
    
    if check_output(vol2):
        mean_vol2, std_vol2 = ( np.mean(vol2[np.where(vol2>0)]), np.std(vol2[np.where(vol2>0)]) )
    
    # STANDARDIZE
    vol1 = np.true_divide( vol1 - mean_vol1, std_vol1 ) if normalize_both else np.true_divide( vol1 - mean_vol2, std_vol2 )
    if check_output(vol2):
        vol2 = np.true_divide( vol2 - mean_vol2, std_vol2 )

    patches_vol1 = cutup(vol1,(16,192,192),(2,1,1))
    if check_output(vol2):
        patches_vol2 = cutup(vol2,(16,192,192),(2,1,1))

    ijk = patches_vol1.shape[0]*patches_vol1.shape[1]*patches_vol1.shape[2]
    
    if check_output(vol2):
        selected_patches = np.empty((ijk,16,192,192,2), dtype='float32')
        selected_patches[:,:,:,:,0] = np.reshape(patches_vol1,(ijk,16,192,192))
        selected_patches[:,:,:,:,1] = np.reshape(patches_vol2,(ijk,16,192,192))
    else:
        selected_patches = np.reshape(patches_vol1,(ijk,16,192,192,1)).astype('float32')

    return selected_patches

"""
Predicts patches of pCT data from patches of UTE data, 
and combines the result by averaging overlapping patches
"""
def predict(model,patches):
    
    # Settings for patch extraction
    h = 16 # Number of slices 
    sh = 2 # Patch stride

    # Container matrices for data and counter for overlapping patches
    predicted_combined = np.zeros((192,192,192))
    predicted_counter = np.zeros((192,192,192))

    # Process a patch at a time
    for p in range(patches.shape[0]):
        from_h = p*sh # Start slice of patch
        predicted = model.predict(np.reshape(patches[p,:,:,:,:],(1,16,192,192,patches.shape[-1]))) # Predict pCT for patch
        predicted[ predicted == np.nan ] = -1 # Can occur, remove so output does not fail, but set to a value that can be searched for
        predicted_combined[from_h:from_h+h,:,:] += np.reshape(predicted,(16,192,192)) # Insert into container
        predicted_counter[from_h:from_h+h,:,:] += 1 # Update counter in area of patch for later average

    predicted_combined = np.divide(predicted_combined,predicted_counter) # Average over overlapping patches
    predicted_combined[ predicted_combined == np.inf ] = 0 # If divide by zero, remove here (should not occur since counter >> 0).
    
    return predicted_combined

"""
Predicts the DeepUTE pCT image
"""
def predict_DeepUTE(ute1,ute2,version='VE11P'):

    # Load model
    if version == 'VE11P':
        model_h5 = '/opt/caai/share/DeepMRAC/models/DeepUTE/DeepUTE_VE11P_model1_TF2.h5' # UPDATE MODEL 01-09-2020]
    elif version == 'VB20P':
        model_h5 = '/opt/caai/share/DeepMRAC/models/DeepUTE/DeepUTE_VB20P_TF2.h5' # CHANGED TO THIS VERSION 06-03-2019]
    else:
        print('Incorrect software version - no model found')
        return None
    
    model = tf.keras.models.load_model(model_h5,compile=False)
    
    # Load all patches
    patches = get_patches_znorm(ute1,ute2,normalize_both=False)
    
    return predict(model,patches)

"""
Predicts the DeepUTE pCT image
"""
def predict_DeepDixon(inphase,outphase,version='VE11P'):

    # Load model
    if version == 'VE11P':
        model_h5 = '/opt/caai/share/DeepMRAC/models/DeepDixon/DeepDixon_VE11P_model1_TF2.h5' # UPDATE MODEL 01-09-2020]
    elif version == 'VB20P':
        model_h5 = '/opt/caai/share/DeepMRAC/models/DeepDixon/DeepDixon_VB20P_TF2.h5' # CHANGED TO THIS VERSION 06-03-2019]
    else:
        print('Incorrect software version - no model found')
        return None
    
    model = tf.keras.models.load_model(model_h5,compile=False)
    
    # Load all patches
    patches = get_patches_znorm(inphase,outphase,normalize_both=True)
    
    return predict(model,patches)

"""
Predicts the DeepT1 pCT image
"""
def predict_DeepT1(t1,version='VE11P'):

    # Load model
    if version == 'VE11P':
        model_h5 = '/opt/caai/share/DeepMRAC/models/DeepT1/DeepT1_VE11P_model1_TF2.h5' # UPDATE MODEL 01-09-2020]
    elif version == 'VB20P':
        model_h5 = '/opt/caai/share/DeepMRAC/models/DeepT1/DeepT1_VB20P_TF2.h5' # CHANGED TO THIS VERSION 06-03-2019]
    else:
        print('Incorrect software version - no model found')
        return None
    
    model = tf.keras.models.load_model(model_h5,compile=False)
    
    # Load all patches
    patches = get_patches_znorm(t1)
    
    return predict(model,patches)
