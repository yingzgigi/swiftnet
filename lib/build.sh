#!/usr/bin/env bash
rm cylib.so

cython -a cylib.pyx -o cylib.cc

#g++ -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing \
#-I/usr/lib/python3.7/site-packages/numpy/core/include -I/usr/include/python3.7m -o cylib.so cylib.cc
g++ -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing \
-I/usr/lib/python3.6/site-packages/numpy/core/include -I/usr/include/python3.6 -o cylib.so cylib.cc
