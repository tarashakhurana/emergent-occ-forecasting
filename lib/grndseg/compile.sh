#!/bin/bash
mkdir build
cd build
cmake -DPYTHON_LIBRARY=/home/tkhurana/anaconda3/envs/ff/lib/libpython3.8.so -DPYTHON_EXECUTABLE=/home/tkhurana/anaconda3/envs/ff/bin/python ..

make
cd ..
