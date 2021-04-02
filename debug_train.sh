#!/bin/bash

CUDA_VISIBLE_DEVICES="" rm -f ~/Data/mol-results/* && python train.py debug.stop_at=10 debug.cb_n_batches=1 batch_size=2
