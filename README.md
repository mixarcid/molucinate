## MOLUCINATE

Here's the source code for MOLUCINATE (MOLecUlar ConvolutIoNal generATive modEl).
How to get this running:

# 1. Install prereqs
First install OpenBabel. Create a conda environment with Python >= 2.8, PyTorch, and RDKit. Then install pip requirements:
```bash
pip install -r requirements-pip.txt
```
# 2. Download ZINC dataset
```bash
./download_zinc.sh ZINC_FOLDER_DESTINATION
```
# 3. Start training
```bash
python train.py
```
