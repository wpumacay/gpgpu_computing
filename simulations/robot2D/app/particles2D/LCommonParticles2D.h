
#pragma once

#define PI 3.1415926

#include <cstdlib>
#include <cmath>

#define RANDOM_SYM( x ) x * 2.0f * ( rand() / ( ( float ) RAND_MAX ) - 0.5 )
#define RANDOM() ( rand() / ( ( float ) RAND_MAX ) )


// #define USE_OPENCL 1
// #define USE_CUDA 1
#define USE_CPU 1

#define PARTICLE_WORLD_SIZE_X 1000.0f
#define PARTICLE_WORLD_SIZE_Y 1000.0f

#define PARTICLE_SPEED_MIN 1.0f
#define PARTICLE_SPEED_MAX 10.0f

#define PARTICLE_SIZE 1.0f

namespace app
{

	namespace particles2D
	{

		struct LPoint
		{
			float x;
			float y;

			int glIndx;
		};

		struct LLine
		{
			LPoint p1;
			LPoint p2;

			int glIndx;
		};

		struct LRectBlock
		{
			LPoint p1;
			LPoint p2;
			LPoint p3;
			LPoint p4;

			int glIndx;
		};

		struct LParticle
		{
			float x;
			float y;

			float vx;
			float vy;

			float r;

			int glIndx;

			void update( float dt )
			{
				x += vx * dt;
				y += vy * dt;
			}

		}


	}






}


