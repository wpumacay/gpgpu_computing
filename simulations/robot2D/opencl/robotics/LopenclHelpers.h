
#pragma once


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <CL/cl.hpp>

#include "../../common.h"

using namespace std;


namespace clUtils
{

    string loadKernelSrc( const string &path );
    
    cl::Program createProgram( const cl::Context &context,
                               const cl::Device &device,
                               const string& pathToKernelSrc );
                               

}




struct ClParticle
{

    float x;
    float y;
    float t;

    float d1;
    float d2;
    float d3;

    float wz;
    float rayZ[NUM_SENSORS];

    ClParticle()
    {
        x = 0;
        y = 0;
        t = 0;

        d1 = 0.0f;
        d2 = 0.0f;
        d3 = 0.0f;

        wz = 1.0f;
    }

    ClParticle( float px, float py, float pt )
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

struct ClLine
{
    float p1x;
    float p1y;
    float p2x;
    float p2y;

    ClLine()
    {
        p1x = 0.0f;
        p1y = 0.0f;
        p2x = 0.0f;
        p2y = 0.0f;
    }

    ClLine( float _p1x, float _p1y, float _p2x, float _p2y )
    {
        p1x = _p1x;
        p1y = _p1y;
        p2x = _p2x;
        p2y = _p2y;
    }
};



void cl_rb_pf_motion_model_step( cl::Program& program,
                                 cl::Context& context,
                                 cl::Device& device,
                                 ClParticle* h_particles, int nParticles, 
                                 float dt, float v, float w );

void cl_rb_pf_sensor_model_step( cl::Program& program,
                                 cl::Context& context,
                                 cl::Device& device,
                                 ClParticle* h_particles, int nParticles, 
                                 ClLine* h_lines, int nLines,
                                 float* h_sensorsZ, float* h_sensorsAng, int nSensors );