
#pragma once

#include <cstdlib>
#include <cmath>

#include "../../common.h"

#define PROB_SAMPLE_SIZE 12

#define RANDOM_SYM( x ) x * 2.0f * ( rand() / ( ( float ) RAND_MAX ) - 0.5 )
#define RAND( x ) x * ( rand() / ( ( float ) RAND_MAX ) )

#define DEFAULT_ALPHA_MOTION_MODEL_1 0.1f
#define DEFAULT_ALPHA_MOTION_MODEL_2 0.1f
#define DEFAULT_ALPHA_MOTION_MODEL_3 0.01f
#define DEFAULT_ALPHA_MOTION_MODEL_4 0.01f
#define DEFAULT_ALPHA_MOTION_MODEL_5 0.001f
#define DEFAULT_ALPHA_MOTION_MODEL_6 0.001f

#include <iostream>

using namespace std;

namespace app
{

	namespace robotics2D
	{

		float sample_normal_distribution( float b );

		struct LPoint
		{
			float x;
			float y;
		};

		struct LLine
		{
			LPoint p1;
			LPoint p2;
		};

		struct LParticle
		{
			float x;
			float y;
			float t;

		    float d1;
		    float d2;
		    float d3;

			int glIndx;

			float wz;

			LParticle()
			{
				x = 0;
				y = 0;
				t = 0;

		        d1 = 0.0f;
		        d2 = 0.0f;
		        d3 = 0.0f;

				wz = 1.0f;
			}

			LParticle( float px, float py, float pt )
			{
				x = px;
				y = py;
				t = pt;

		        d1 = 0.0f;
		        d2 = 0.0f;
		        d3 = 0.0f;

				wz = 1.0f;
			}

			void update( float dt,
						 float v, float w,
						 float alpha1 = DEFAULT_ALPHA_MOTION_MODEL_1, 
						 float alpha2 = DEFAULT_ALPHA_MOTION_MODEL_2, 
						 float alpha3 = DEFAULT_ALPHA_MOTION_MODEL_3, 
						 float alpha4 = DEFAULT_ALPHA_MOTION_MODEL_4,
						 float alpha5 = DEFAULT_ALPHA_MOTION_MODEL_5, 
						 float alpha6 = DEFAULT_ALPHA_MOTION_MODEL_6 )
			{
				v = v + sample_normal_distribution( alpha1 * abs( v ) +
													alpha2 * abs( w ) );

				w = w + sample_normal_distribution( alpha3 * abs( v ) +
													alpha4 * abs( w ) );

				float rt = sample_normal_distribution( alpha5 * abs( v ) +
												 	   alpha6 * abs( w ) );

				if ( abs( w ) < 0.001f )
				{
					this->x += v * dt * cos( t );
					this->y += v * dt * sin( t );
				}
				else
				{
					this->x += ( v / w ) * ( sin( this->t + w * dt ) - sin( this->t ) );
					this->y += ( v / w ) * ( -cos( this->t + w * dt ) + cos( this->t ) );
					this->t += w * dt + rt * dt;
				}
				
			}
		};



	}

}