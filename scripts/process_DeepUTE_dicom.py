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

import argparse, os, shutil, datetime
from rhscripts.DeepMRAC import predict_DeepUTE
import numpy as np
import pydicom as dicom  
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
Check that the correct number of files is present, and load the UTE TE1 and TE2 images
"""
def load_data():
    utes = [ f for f in os.listdir('%s' % tmpdir) if not f.startswith('.') ]
    utes.sort(key=float)
    
    # Check that correct number of files is present
    assert len(os.listdir('%s/%s' % (tmpdir, utes[0]))) == 192
    assert len(os.listdir('%s/%s' % (tmpdir, utes[1]))) == 192
    assert len(os.listdir('%s/%s' % (tmpdir, utes[2]))) == 192
    
    # Load UTE TE1
    ute1 = np.empty((192,192,192))
    for filename in os.listdir('%s/%s' % (tmpdir, utes[0])):
        ds = dicom.dcmread('%s/%s/%s' % (tmpdir, utes[0], filename))
        i = int(ds.InstanceNumber)-1
        ute1[i,:,:] = ds.pixel_array
        
    # Load UTE TE2
    ute2 = np.empty((192,192,192))
    for filename in os.listdir('%s/%s' % (tmpdir, utes[1])):
        ds = dicom.dcmread('%s/%s/%s' % (tmpdir, utes[1], filename))
        i = int(ds.InstanceNumber)-1
        ute2[i,:,:] = ds.pixel_array
        
    return ute1,ute2
    
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
    
    utes = [ f for f in os.listdir('%s' % tmpdir) if not f.endswith('.mnc') or f.startswith('.') ]
    utes.sort(key=float)
    
    # Determine where to store the output dicom files
    if folder_outname == None:
       outname = os.path.join(patient,'DeepUTE')
    else:
       outname = folder_outname
    
    # Convert the files
    to_dcm(DeepX,os.path.join(tmpdir,utes[2]),outname)

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

    parser = argparse.ArgumentParser(description='Predict using DeepUTE.')
    parser.add_argument("patient", help="Path to patient.")
    parser.add_argument("--outname", help="Name for output folder. ", type=str)
    parser.add_argument("--version", help="Software version used to train the model (VB20P or VE11P) Default: VE11P. ", type=str, default='VE11P')
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Create temporary folder    
    tmpdir = tempfile.mkdtemp()

    # Sort files into specific folders
    sort_files(args.patient)
    
    # Load data
    ute1, ute2 = load_data()
    
    # Predict
    if verbose:
        print("Predicting DeepUTE using %s model" % args.version)
    DeepX = predict_DeepUTE(ute1,ute2,args.version)
    if not check_output(DeepX):
        exit(-1)
    
    # Convert to DICOM
    convert_to_DCM(DeepX,args.patient,args.outname)
    
    # Cleanup
    shutil.rmtree(tmpdir)
    if os.path.exists('inphase_flirt.mat'):
        os.remove('inphase_flirt.mat')
        os.remove('opposedphase_flirt.mat')
        os.remove('DeepDixon_flirt.mat')
    
    
    
