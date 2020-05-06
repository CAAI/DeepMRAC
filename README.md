# DeepMRAC
Deep learning based peudoCT generation from MRI.

DeepMRAC is a deep learning network for obtaining attenuation correction umaps. The network currently works with input images:
 - Dixon-VIBE
 - UTE

All versions are implemented for VB20P and VE11P in seperate models.

Current realease only includes VE11P models. Future release will also include VB20P and models for T1w MPRAGE. 

## Requirements
Install appropriate GPU, CUDA and cuDNN toolkits. See https://www.tensorflow.org/install/gpu.

To install the required software tools using pip, run:

```
pip install -r requirements.txt
```

Please note - to use DeepDixon, you further need to install FSL.

The runtime with a decent GPU (e.g. Titan V) is about 4 seconds. The running time is about 15-20 minutes for CPU.

## Installation
To install the scripts and models:
``
python install.py
``

Installation will place the run scripts at /opt/caai/bin/ and /opt/caai/rhscripts/, and the models at /opt/caai/share/.
The folders will be automatically created. Change the paths in the install.py script and DeepMRAC.py script if you wish to install elsewhere.

DeepMRAC are written for Tensorflow >= 2. The models have been tested with Tensorflow 2.1 as well as Tensorflow 1.8 (with Keras 2.2.4) on Ubuntu 18.04 running TITAN V and RTX.

## Running the script

### Using DICOM input data
```
process_DeepUTE_dicom.py ﹤path to DICOM data﹥
process_DeepDixon_dicom.py ﹤path to DICOM data﹥
```

The output will be a folder called DeepUTE/DeepDixon within the DICOM data folder.

### Using python function ( with pre-loaded data )
```python
from rhscripts.DeepMRAC import predict_DeepUTE, predict_DeepDixon
pseudoCT_UTE = predict_DeepUTE(ute1,ute2)
pseudoCT_Dixon = predict_DeepDixon(inphase,outphase)
```

## Contact
Claes Ladefoged, Rigshospitalet, Copenhagen, Denmark
claes.noehr.ladefoged@regionh.dk

## Citation
The publication has been submitted for review - please contact claes.noehr.ladefoged@regionh.dk for details on citations in the mean time