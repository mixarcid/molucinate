# MOLUCINATE

Here's the source code for MOLUCINATE (MOLecUlar ConvolutIoNal generATive modEl).
How to get this running:

## 1. Install prereqs
First install OpenBabel. Create a conda environment with Python >= 2.8, PyTorch, and RDKit. Then install pip requirements:
```bash
pip install -r requirements-pip.txt
```
## 2. Download ZINC dataset
```bash
./download_zinc.sh ZINC_FOLDER_DESTINATION
```
## 3. Create local configuration file
Create a file `cfg/platform/local.yaml'. This contains all the configuration that should differ per-computer. Add in this information:
```yaml
# @package _group_

num_workers: NUM_DATALOADING_WORKERS

zinc_dir: ZINC_FOLDER_DESTINATION # from above

results_path: PATH_TO_RESULTS
```

## 4. Start training
```bash
python train.py
```
