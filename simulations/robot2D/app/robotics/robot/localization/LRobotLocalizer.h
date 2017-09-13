
#pragma once

#include "../../LCommonRobotics2D.h"
#include "../LRobot2D.h"

#include <vector>

using namespace std;

#define NUM_PARTICLES 100

namespace app
{

	namespace robotics2D
	{

		class LRobot2D;

		class LRobotLocalizer
		{

			private :

			LRobot2D* m_parent;
			vector<LParticle*> m_particles;
			vector<int> m_indxs;

			public :

			LRobotLocalizer( LRobot2D* parent );
			~LRobotLocalizer();

			void update( float dt );

		};




	}



}