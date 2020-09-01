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

## Updating the script
Download the latest version of the code, delete the old models and run installation again.
```
git pull
rm -rf models.zip /opt/caai/share/DeepAC
python install.py
```

## Running the script

### Using DICOM input data
```
process_DeepUTE_dicom.py ﹤path to DICOM data﹥
process_DeepDixon_dicom.py ﹤path to DICOM data﹥
process_DeepT1_dicom.py < path to DICOM data >
```

The output will be a folder called DeepUTE/DeepDixon/DeepT1 within the DICOM data folder.

### Using python function ( with pre-loaded data )
**NOTE** The data for Dixon (inphase and outphase) and T1 must be preprocessed to isotropic voxel size on a 192x192 matrix. See the process_X_dicom.py scripts for details.

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
The models were trained on a *POWER AC922* computer with **4 NVIDIA Tesla V100 32GB**. The computer from IBM allowed an increased batch size used during training (12 vs 3 previously used).

## Contact
Claes Ladefoged, Rigshospitalet, Copenhagen, Denmark
claes.noehr.ladefoged@regionh.dk

## Citation
Please cite the main method manuscript when using our method.

Ladefoged CN, Hansen AE, Henriksen OM, et al. AI-driven attenuation correction for brain PET/MRI: Clinical evaluation of a dementia cohort and importance of the training group size. Published online ahead of print, 2020 Aug 1. Neuroimage. 2020;222:117221. [doi:10.1016/j.neuroimage.2020.117221](https://www.sciencedirect.com/science/article/pii/S1053811920307072)

DeepUTE has been previously evaluated in the following publications:

Ladefoged CN, Marner L, Hindsholm A, Law I, Højgaard L, Andersen FL. Deep Learning Based Attenuation Correction of PET/MRI in Pediatric Brain Tumor Patients: Evaluation in a Clinical Setting. Front Neurosci. 2019;12:1005. Published 2019 Jan 7. [doi:10.3389/fnins.2018.01005](https://www.frontiersin.org/articles/10.3389/fnins.2018.01005/full)

Øen SK, Keil TM, Berntsen EM, et al. Quantitative and clinical impact of MRI-based attenuation correction methods in 18F-FDG evaluation of dementia. EJNMMI Res. 2019;9(1):83. Published 2019 Aug 24. [doi:10.1186/s13550-019-0553-2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6708519/)

