#!/bin/bash

rm -f ~/Data/mol-results/* && CUDA_VISIBLE_DEVICES="" python train.py debug.stop_at=10 debug.cb_n_batches=1 batch_size=2
