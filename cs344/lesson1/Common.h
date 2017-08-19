
#pragma once


#define USE_CUDA 1
//#define USE_OPENCL 1

typedef unsigned char u8;

#ifdef USE_OPENCL


struct uchar4
{
    u8 x;
    u8 y;
    u8 z;
    u8 w;
};

#endif
