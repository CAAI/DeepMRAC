# DeepMRAC
Deep learning based peudoCT generation from MRI

DeepMRAC is a deep learning network for obtaining attenuation correction umaps. The network currently works with input images:
 - Dixon-VIBE
 - UTE

All versions are implemented for VB20P and VE11P in seperate models.

Future release will also include T1w MPRAGE. 

## Requirements
First, if you need to use GPU, you need to install the GPU, CUDA and cuDNN toolkits. See https://www.tensorflow.org/install/gpu.

To install the required software tools, run:
``
pip3 install -r requirements.txt
``

Please note - to use DeepDixon, you further need to install FSL.

The best performance will be obtained with GPU support (about 4 seconds, depending on GPU type), but CPU can also be used. Please install tensorflow rather than tensorflow-gpu (in requirements). The running time is about 15-20 minutes for CPU.

## Installation
Simply run the command below to install the scripts, download the models and install them. Installation directory will be at /opt/caai/bin/ and /opt/caai/share.
Change the paths in the install.py script if you wish to install elsewhere.

``
python install.py
``

## Running the script

### Using python function ( with pre-loaded data )
``
from rhscripts.DeepMRAC import predict_DeepUTE, predict_DeepDixon
``

### Using DICOM input data
``
process_DeepUTE_dicom.py ﹤path to DICOM data﹥
``

or 

``
process_DeepUTE_dicom.py ﹤path to DICOM data﹥
``

The output will be a folder called DeepUTE/DeepDixon within the DICOM data folder.

### Using python function ( with pre-loaded data )
``
from rhscripts.DeepMRAC import predict_DeepUTE, predict_DeepDixon

pseudoCT_UTE = predict_DeepUTE(ute1,ute2)

pseudoCT_Dixon = predict_DeepUTE(inphase,outphase)
``

## Contact
Claes Ladefoged, Rigshospitalet, Copenhagen, Denmark
claes.noehr.ladefoged@regionh.dk

## Citation
The publication has been submitted for review - please contact claes.noehr.ladefoged@regionh.dk for details on citations in the mean time