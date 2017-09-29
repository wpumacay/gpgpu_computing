
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cmath>

#include "../../common.h"

#define RB_PROB_SAMPLE_SIZE 12

#define RB_ALPHA_MOTION_MODEL_1 0.05f
#define RB_ALPHA_MOTION_MODEL_2 0.05f
#define RB_ALPHA_MOTION_MODEL_3 0.005f
#define RB_ALPHA_MOTION_MODEL_4 0.005f
#define RB_ALPHA_MOTION_MODEL_5 0.0005f
#define RB_ALPHA_MOTION_MODEL_6 0.0005f

#define CU_THREADS_PER_BLOCK 128

//#define CU_USE_3D_SEPARATION 1

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

struct CuParticle
{

    float x;
    float y;
    float t;

    float d1;
    float d2;
    float d3;

    float wz;
    float rayZ[NUM_SENSORS];

    __host__ __device__ CuParticle()
    {
        x = 0;
        y = 0;
        t = 0;

        d1 = 0.0f;
        d2 = 0.0f;
        d3 = 0.0f;

        wz = 1.0f;
    }

    __host__ __device__ CuParticle( float px, float py, float pt )
    {
        x = px;
        y = py;
        t = pt;

        d1 = 0.0f;
        d2 = 0.0f;
        d3 = 0.0f;

        wz = 1.0f;
    }
    
};

struct CuLine
{
    float p1x;
    float p1y;
    float p2x;
    float p2y;

    __host__ __device__ CuLine()
    {
        p1x = 0.0f;
        p1y = 0.0f;
        p2x = 0.0f;
        p2y = 0.0f;
    }

    __host__ __device__ CuLine( float _p1x, float _p1y, float _p2x, float _p2y )
    {
        p1x = _p1x;
        p1y = _p1y;
        p2x = _p2x;
        p2y = _p2y;
    }
};

/*
For check only, declared and defined in the .cu related file
__global__ void rb_init_random_generator( u32 seed,
                                          curandState_t* pStates,
                                          int nParticles );
*/

__device__ float rb_sample_normal_distribution( int kIdx, curandState_t* pCurandStates,
                                                float sigma2 );

/*
For check only, declared and defined in the .cu related file
__global__ void rb_update_particle( CuParticle* pParticles, int nParticles,
                                    curandState_t* pCurandStates,
                                    float dt, float v, float w );
*/

void rb_pf_motion_model_step( CuParticle* h_particles, int nParticles, 
                              float dt, float v, float w );

void rb_pf_sensor_model_step( CuParticle* h_particles, int nParticles,
                              CuLine* h_lines, int nLines,
                              float* h_sensorsZ, float* h_sensorsAng, int nSensors );