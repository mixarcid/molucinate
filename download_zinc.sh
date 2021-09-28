#!/bin/bash
echo "extracting ZINC to $1"
mkdir -p $1
wget http://zinc15.docking.org/substances/subsets/for-sale.mol2?count=all -O $1/all.mol2
obabel -imol2 $1/all.mol2 -omol2 -O$1/zinc.mol2 -m
python preprocess_zinc.py $1
python filter_zinc.py
