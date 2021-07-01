#!/bin/bash

rm -f ~/Data/mol-results/* && CUDA_VISIBLE_DEVICES="0" python train.py debug.stop_at=10 debug.cb_n_batches=1 debug.checkpoint_n_batches=1 batch_size=2 $@
