#!/bin/bash

#  -m cProfile -o train.prof
rm -f ~/Data/mol-results/*.png &&
  #CUDA_VISIBLE_DEVICES="0" python train.py debug.stop_at=10000 debug.cb_n_batches=null debug.checkpoint_n_batches=null batch_size=4 debug.profile=true $@
  py-spy top -- python train.py debug.stop_at=1000 debug.cb_n_batches=null debug.checkpoint_n_batches=null batch_size=4 debug.profile=true
