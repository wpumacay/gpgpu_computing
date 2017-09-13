
#include "LCommonRobotics2D.h"

namespace app
{
	namespace robotics2D
	{
		float sample_normal_distribution( float b )
		{
			float _res = 0.0f;
			for ( int q = 0; q < PROB_SAMPLE_SIZE; q++ )
			{
				_res += RANDOM_SYM( b );
			}

			return 0.5f * _res;
		}

		
	}
}