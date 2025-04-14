#!/usr/bin/bash
nvcc --std=c++17 --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage --use_fast_math -I../../../3rdparty/cutlass/include/ -I../../../src/ -O3 --compiler-options '-fPIC' --shared sparse_decoding.cu -lcuda  -gencode=arch=compute_80,code=sm_80 -o sparse_decoding.so 

