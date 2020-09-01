# DeepMRAC
Deep learning based peudoCT generation from MRI.

![Example pseudoCT images](/images/figure2.png)

DeepMRAC is a deep learning network for obtaining attenuation correction umaps. The network works with input images:
 - Dixon-VIBE
 - T1 weighted MPRAGE
 - Ultra-short echo time (UTE)

All versions are implemented for VB20P and VE11P in seperate models.

## Requirements
Install appropriate GPU, CUDA and cuDNN toolkits. See https://www.tensorflow.org/install/gpu.

To install the required software tools using pip, run:

```
pip install -r requirements.txt
```

Please note - to use DeepDixon or DeepT1, you further need to install FSL, as the images are preprocessed to correct resolution.

The runtime with a decent GPU (e.g. Titan V) is about 4 seconds. The running time is about 15-20 minutes for CPU.

## Installation
To install the scripts and models:

``
python install.py
``

Installation will place the run scripts at /opt/caai/bin/ and /opt/caai/rhscripts/, and the models at /opt/caai/share/. Make sure to have write access to /opt.
The folders will be automatically created. Change the paths in the install.py script and DeepMRAC.py script if you wish to install elsewhere.

Add /opt/caai/ to PYTHONPATH and /opt/caai/bin to PATH to use the scripts from anywhere.

Example:

```
export PATH=$PATH:/opt/caai/bin
export PYTHONPATH=$PYTHONPATH:/opt/caai
```

DeepMRAC are written for Tensorflow >= 2. The models have been tested with Tensorflow 2.1 as well as Tensorflow 1.8 (with Keras 2.2.4) on Ubuntu 18.04 running TITAN V and RTX.

## Running the script

### Using DICOM input data
```
process_DeepUTE_dicom.py ﹤path to DICOM data﹥
process_DeepDixon_dicom.py ﹤path to DICOM data﹥
process_DeepT1_dicom.py < path to DICOM data >
```

The output will be a folder called DeepUTE/DeepDixon/DeepT1 within the DICOM data folder.

### Using python function ( with pre-loaded data )
```python
from rhscripts.DeepMRAC import predict_DeepUTE, predict_DeepDixon, predict_DeepT1
pseudoCT_UTE = predict_DeepUTE(ute1,ute2)
pseudoCT_Dixon = predict_DeepDixon(inphase,outphase)
pseudoCT_T1 = predict_DeepT1(t1)
```

## How the models were trained
### Patients
The models were trained solely using Siemens Biograph mMR data from two software versions (VB20P and VE11P). We expect that DeepT1 should work for T1w MPRAGE sequences from other scanners, but this was not thoroughly tested. We require that the input images are aligned and resampled the same way the models were trained when the models are used for inference.

The VB20P models was trained and validated using **800+ subjects**. The VE11P models are fine-tuned from the VB20P models using **200+ subjects**.

### Hardware
The models were trained in two rounds.

The first set of models were trained using NVIDIA Titan V or Titan RTX GPU on a desktop computer running Ubuntu 18.04.

The second set of models were trained after a collaboration with **IBM** was established. 
The models were trained on a *POWER AC922* computer with **2 NVIDIA Tesla V100 16GB**. The computer from IBM allowed an increased batch size used during training (12 vs 3 previously used).

## Contact
Claes Ladefoged, Rigshospitalet, Copenhagen, Denmark
claes.noehr.ladefoged@regionh.dk

## Citation
The publication has been submitted for review - please contact claes.noehr.ladefoged@regionh.dk for details on citations in the mean time
