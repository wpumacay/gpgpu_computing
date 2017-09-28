
#pragma once

#define PI 3.1415926

#include <cstdlib>
#include <cmath>

#define RANDOM_SYM( x ) x * 2.0f * ( rand() / ( ( float ) RAND_MAX ) - 0.5 )
#define RANDOM() ( ( (float)rand() ) / ( ( float ) RAND_MAX ) )


// #define USE_OPENCL 1
// #define USE_CUDA 1
#define USE_CPU 1

#define PARTICLE_WORLD_SIZE_X 1000.0f
#define PARTICLE_WORLD_SIZE_Y 1000.0f

#define PARTICLE_SPEED_MIN 100.0f
#define PARTICLE_SPEED_MAX 200.0f

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

			// Normals
			float nx;
			float ny;

			// Udir
			float ux;
			float uy;

			float len;

			int glIndx;

			LLine( float p1x, float p1y,
				   float p2x, float p2y,
				   bool inPoint = false,
				   float inX = 0, float inY = 0 )
			{
				p1.x = p1x;
				p1.y = p1y;
				p2.x = p2x;
				p2.y = p2y;

				computeNormals( inPoint, inX, inY );
			}

			void computeNormals( bool inPoint, float inX, float inY )
			{
				// calculate udir from p1 to p2
				float dx = p2.x - p1.x;
				float dy = p2.y - p1.y;
				len = sqrt( dx * dx + dy * dy );

				ux = dx / len;
				uy = dy / len;

				nx = -uy;
				ny = ux;

				// If inner part should point to somewhere, point normal to it
				if ( inPoint )
				{
					float ddx = inX - p1.x;
					float ddy = inY - p1.y;

					float ddot = nx * ddx + ny * ddy;
					// if pointing in the wrong direction, flip
					if ( ddot < 0 )
					{
						nx *= -1;
						ny *= -1;
					}
				}
			}
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

		};


	}






}


