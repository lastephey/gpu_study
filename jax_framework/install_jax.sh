#!/usr/bin/env bash

# install jaxlib
PYTHON_VERSION=cp38  # alternatives: cp36, cp37, cp38
CUDA_VERSION=cuda102  # alternatives: cuda92, cuda100, cuda101, cuda102
#PLATFORM=linux_x86_64  # alternatives: linux_x86_64
PLATFORM=manylinux2010_x86_64  # alternatives: manylinux2010_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.52-$PYTHON_VERSION-none-$PLATFORM.whl

pip install --upgrade jax  # install jax
