
#pragma once

#include "../../LCommonRobotics2D.h"
#include "../LRobot2D.h"

#include <vector>

using namespace std;

#ifdef USE_CUDA

#include "../../../../cuda/robotics/LRoboticsCudaHelpers.h"

#elif defined( USE_OPENCL )

#include "../../../../opencl/robotics/LopenclHelpers.h"

#endif

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

			#ifdef USE_CUDA

			CuParticle* m_hParticles;

			float* m_hSensorsZ;
			float* m_hSensorsAng;

			CuLine* m_hLines;
			int m_hNumLines;

			#elif defined( USE_OPENCL )

			ClParticle* m_hParticles;

			float* m_hSensorsZ;
			float* m_hSensorsAng;

			ClLine* m_hLines;
			int m_hNumLines;

			cl::Context m_context;
			cl::Device m_device;
			cl::Platform m_platform;
			cl::Program m_program;

			#endif

			int calcNumParticlesInRange( float px, float py, float dRange );

			public :

			LRobotLocalizer( LRobot2D* parent );
			~LRobotLocalizer();

			void onMapLoaded( vector<LLine> wallLines );

			void update( float dt, vector<LLine> vMapWalls );

			// particle filter helper methods
			float probSensorModel( const LParticle& pParticle, 
								   float zExp, float sensAngle,
								   const vector<LLine>& vMapWalls );

			void setFilterState( int isEnabled )
			{
				m_useFilter = isEnabled;
			}

			void reset();

			void dumpInfo();

			void dumpParticles();
		};




	}



}