#!/usr/bin/env python3

import wget
from pathlib import Path
import zipfile
from shutil import copyfile, copymode
import os

# Setup folders
if not os.path.exists('/opt/caai/share'):
	print("Setting up install directories at /opt/caai")
	Path('/opt/caai/share').mkdir(parents=True, exist_ok=True)
	Path('/opt/caai/bin').mkdir(parents=True, exist_ok=True)

# Download models
if not os.path.exists('models.zip'):
	print('Downloading models for DeepMRAC')
	url = "http://resolute.pet.rh.dk:8000/models.zip"
	wget.download(url,'models.zip')
	print("")

# Unzip models
if not os.path.exists('/opt/caai/share/DeepMRAC'):
	print("Extracting models")
	with zipfile.ZipFile('models.zip', 'r') as zip_ref:
		zip_ref.extractall('/opt/caai/share/DeepMRAC')

# Install scripts
if not os.path.exists('/opt/caai/bin/process_DeepDixon_dicom.py'):
	print("Installing run scripts")	
	copyfile('scripts/process_DeepDixon_dicom.py', '/opt/caai/bin/process_DeepDixon_dicom.py')
	copyfile('scripts/process_DeepUTE_dicom.py', '/opt/caai/bin/process_DeepUTE_dicom.py')
	copymode('scripts/process_DeepDixon_dicom.py', '/opt/caai/bin/process_DeepDixon_dicom.py')
	copymode('scripts/process_DeepUTE_dicom.py', '/opt/caai/bin/process_DeepUTE_dicom.py')

