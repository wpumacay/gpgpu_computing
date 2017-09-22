
#pragma once

#define PI 3.1415926

#include <cstdlib>
#include <cmath>

#define PROB_SAMPLE_SIZE 12

#define RANDOM_SYM( x ) x * 2.0f * ( rand() / ( ( float ) RAND_MAX ) - 0.5 )

#define DEFAULT_ALPHA_MOTION_MODEL_1 0.1f
#define DEFAULT_ALPHA_MOTION_MODEL_2 0.1f
#define DEFAULT_ALPHA_MOTION_MODEL_3 0.01f
#define DEFAULT_ALPHA_MOTION_MODEL_4 0.01f
#define DEFAULT_ALPHA_MOTION_MODEL_5 0.001f
#define DEFAULT_ALPHA_MOTION_MODEL_6 0.001f

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

			LParticle( float px, float py, float pt )
			{
				x = px;
				y = py;
				t = pt;
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
					x += v * dt * cos( t );
					y += v * dt * sin( t );
				}
				else
				{
					x += ( v / w ) * ( sin( t + w * dt ) - sin( t ) );
					y += ( v / w ) * ( -cos( t + w * dt ) + cos( t ) );
					t += w * dt + rt * dt;
				}
					

			}
		};



	}

}