
#pragma once

#include "../../LCommonRobotics2D.h"
#include "../LRobot2D.h"

#include <vector>

using namespace std;

#define DRAW_PARTICLES 1000
#define NUM_PARTICLES  1000
#define SIGMA_SENSOR 20.0

namespace app
{

	namespace robotics2D
	{

		class LRobot2D;

		class LRobotLocalizer
		{

			private :

			LRobot2D* m_parent;
			LParticle m_particles[NUM_PARTICLES];
			vector<int> m_glIndxs;

			float m_particleWeights[NUM_PARTICLES];
			float m_cdfParticleWeights[NUM_PARTICLES];
			int m_resampleIndxs[NUM_PARTICLES];

			int m_useFilter;

			public :

			LRobotLocalizer( LRobot2D* parent );
			~LRobotLocalizer();

			void update( float dt, vector<LLine> vMapWalls );

			// particle filter helper methods
			float probSensorModel( const LParticle& pParticle, 
								   float zExp, float sensAngle,
								   const vector<LLine>& vMapWalls );

			void setFilterState( int isEnabled )
			{
				m_useFilter = isEnabled;
			}

			void dumpParticles();
		};




	}



}