
#pragma once

#include "../../../gl/core/base/LBaseObject2D.h"
#include "../../../gl/core/primitives/LPrimitivePoint.h"
#include "../../../gl/core/primitives/LPrimitivesRenderer2D.h"
#include <cmath>
#include <vector>

using namespace std;

#include "LRobotLaserSensor.h"
#include "../LCommonRobotics2D.h"
#include "localization/LRobotLocalizer.h"
#include "LLinePath.h"

#define R_KEY_W 0
#define R_KEY_A 1
#define R_KEY_S 2
#define R_KEY_D 3

#define R_MANUAL_V 200.0f
#define R_MANUAL_W 2.0f * PI / 2.5f

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

			int m_manualControls[4];
			bool m_isInAutonomousMode;

			vector<LRobotLaserSensor*> m_sensors;

			LRobotLocalizer* m_localizer;

			int m_useLocalizerFilter;

			LLine m_errLine;

			float m_lastErrD;
			float m_lastErrX;
			float m_lastErrY;

			float kp;
			float kd;
			float ki;

			float kwp;
			float kwd;
			float kwi;

			float err_now;
			float err_bef;

			float ei;
			float ep;
			float ed;

			float ewp;
			float ewi;
			float ewd;

			float m_enabled;

			public :


			LRobot2D( float x, float y );

			void update( float dt, vector<LLine> vMapWalls, LLinePath* pPath );

			void reset( float x, float y, float theta );

			void onMapLoaded( vector<LLine> wallLines );

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

			vector<LRobotLaserSensor*> sensors();

			void onKeyDown( int pKey );
			void onKeyUp( int pKey );
			void toogleFilter();
		};


	}


}