
#pragma once


//#define USE_CUDA 1
#define USE_OPENCL 1

typedef unsigned char u8;

#ifdef USE_OPENCL

#include <CL/cl.hpp>

struct uchar4
{
    u8 x;
    u8 y;
    u8 z;
    u8 w;
};

#endif

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#endif


// A General context-like container for both CL and CUDA


struct CEnvironment
{

    #ifdef USE_OPENCL

    cl::Platform platform;
    cl::Device device;
    cl::Context context;

    #elif USE_CUDA

    #endif


};


