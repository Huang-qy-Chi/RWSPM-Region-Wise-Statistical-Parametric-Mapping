#!/bin/bash
# ============================================================
# Compile cball_ext.so for Linux
# Requires: gcc
#
# NOTE: If you have libgomp (OpenMP), add -fopenmp to both
# compile and link commands for parallel speedup.
# ============================================================
echo "Compiling cball_ext.so ..."

gcc -shared -O3 -fPIC \
    cball_wrapper.c bdd_matrix.c utilities.c median.c \
    -I. -lm \
    -o cball_ext.so

if [ $? -eq 0 ]; then
    echo "SUCCESS: cball_ext.so created"
    ls -lh cball_ext.so
else
    echo "FAILED: compilation error"
fi
