#!/usr/bin/bash

# Usage: ./compile.sh sparse_decoding 80

set -e

NAME=$1
ARCH=$2

if [ -z "$NAME" ] || [ -z "$ARCH" ]; then
  echo "Usage: $0 <base_name> <compute_arch>"
  echo "Example: $0 sparse_decoding 80"
  exit 1
fi

# If ARCH is 90, change it to 90a
if [ "$ARCH" = "90" ]; then
  ARCH="90a"
fi

nvcc --std=c++17 \
     --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage \
     --use_fast_math \
     -I../../../3rdparty/cutlass/include/ \
     -I../../../src/ \
     -O3 \
     --compiler-options '-fPIC' \
     --shared ${NAME}.cu \
     -lcuda \
     -gencode=arch=compute_${ARCH},code=sm_${ARCH} \
     -o ${NAME}.so
