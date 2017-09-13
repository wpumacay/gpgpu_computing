
#pragma once


#include "LRobot2D.h"
#include "../../../gl/core/primitives/LPrimitivesRenderer2D.h"
#include <cmath>

#define MAX_LEN 1000.0f


namespace app
{

	namespace robotics2D
	{

		class LRobot2D;

		class LRobotLaserSensor
		{
			private :

			LRobot2D* m_parent;
			float m_angle;
			float m_len;
			int m_pIndx;

			public :

			LRobotLaserSensor( LRobot2D* parent, float angle );

			void update( float dt );

			void rayCast();

		};



	}



}