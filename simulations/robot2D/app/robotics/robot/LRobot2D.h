
#pragma once

#include "../../../gl/core/base/LBaseObject2D.h"
#include "../../../gl/core/primitives/LPrimitivePoint.h"
#include <cmath>
#include <vector>

using namespace std;

#include "LRobotLaserSensor.h"
#include "../LCommonRobotics2D.h"
#include "localization/LRobotLocalizer.h"

#define NUM_SENSORS 10


namespace app
{

	namespace robotics2D
	{

		class LRobotLaserSensor;
		class LRobotLocalizer;

		class LRobot2D : public engine::gl::LBaseObject2D
		{

			
			private :

			float m_x;
			float m_y;
			float m_theta;
			float m_v;
			float m_w;

			vector<LRobotLaserSensor*> m_sensors;

			LRobotLocalizer* m_localizer;

			public :

			LRobot2D( float x, float y );

			void update( float dt );

			void setV( float v );
			void setW( float w );

			void setX( float x );
			void setY( float y );
			void setTheta( float theta );

			float getX();
			float getY();
			float getTheta();

			float getV();
			float getW();
		};


	}


}