
#include "LRobotLocalizer.h"
#include "../LRobotLaserSensor.h"
#include "../../../../gl/core/primitives/LPrimitivesRenderer2D.h"

#include <iostream>
#include <cmath>

using namespace std;

namespace app
{

	namespace robotics2D
	{


		LRobotLocalizer::LRobotLocalizer( LRobot2D* parent )
		{
			m_parent = parent;

			for ( int q = 0; q < NUM_PARTICLES; q++ )
			{
				m_particles[q].x = m_parent->getX();
				m_particles[q].y = m_parent->getY();
				m_particles[q].t = m_parent->getTheta();
			}

			for ( int q = 0; q < DRAW_PARTICLES; q++ )
			{
				m_glIndxs.push_back( engine::gl::LPrimitivesRenderer2D::instance->addPoint( m_particles[q].x, 
																							m_particles[q].y,
																							1.0f, 0.0f, 0.0f ) );
			}

			#ifdef USE_CUDA

			m_hParticles = new CuParticle[NUM_PARTICLES];
			m_hSensorsZ = new float[NUM_SENSORS];
			m_hSensorsAng = new float[NUM_SENSORS];

			#endif
		}

		LRobotLocalizer::~LRobotLocalizer()
		{
			
		}

		void LRobotLocalizer::onMapLoaded( vector<LLine> wallLines )
		{
			#ifdef USE_CUDA

			m_hNumLines = wallLines.size();
			m_hLines = new CuLine[m_hNumLines];

			for ( int q = 0; q < m_hNumLines; q++ )
			{
				m_hLines[q].p1x = wallLines[q].p1.x;
				m_hLines[q].p1y = wallLines[q].p1.y;
				m_hLines[q].p2x = wallLines[q].p2.x;
				m_hLines[q].p2y = wallLines[q].p2.y;
			}

			#endif
		}

		void LRobotLocalizer::update( float dt, vector<LLine> vMapWalls )
		{

			float vv = m_parent->getV();
			float ww = m_parent->getW();

			#ifdef USE_CUDA

				for ( int q = 0; q < NUM_PARTICLES; q++ )
				{
					m_particles[q].d1 = sample_normal_distribution( DEFAULT_ALPHA_MOTION_MODEL_1 * abs( vv ) +
																	DEFAULT_ALPHA_MOTION_MODEL_2 * abs( ww ) );
					m_particles[q].d2 = sample_normal_distribution( DEFAULT_ALPHA_MOTION_MODEL_3 * abs( vv ) +
																	DEFAULT_ALPHA_MOTION_MODEL_4 * abs( ww ) );
					m_particles[q].d3 = sample_normal_distribution( DEFAULT_ALPHA_MOTION_MODEL_5 * abs( vv ) +
																	DEFAULT_ALPHA_MOTION_MODEL_6 * abs( ww ) );

					m_hParticles[q].x = m_particles[q].x;
					m_hParticles[q].y = m_particles[q].y;
					m_hParticles[q].t = m_particles[q].t;
					m_hParticles[q].d1 = m_particles[q].d1;
					m_hParticles[q].d2 = m_particles[q].d2;
					m_hParticles[q].d3 = m_particles[q].d3;
					for ( int s = 0; s < NUM_SENSORS; s++ )
					{
						m_hParticles[q].rayZ[s] = MAX_LEN;
					}
					m_hParticles[q].wz = MAX_LEN;
				}

				rb_pf_motion_model_step( m_hParticles, NUM_PARTICLES,
										 dt, vv, ww );

				for ( int q = 0; q < NUM_PARTICLES; q++ )
				{
					m_particles[q].x = m_hParticles[q].x;
					m_particles[q].y = m_hParticles[q].y;
					m_particles[q].t = m_hParticles[q].t;
				}

			#else

				for ( int q = 0; q < NUM_PARTICLES; q++ )
				{
					// update each particle based on the motion model
					m_particles[q].update( dt, m_parent->getV(), m_parent->getW() );
				}

			#endif

			if ( m_useFilter )
			{
				// particle filter algorithm **************************************************
				vector<LRobotLaserSensor*> vSensors = m_parent->sensors();

				#ifdef USE_CUDA
				//cout << "using filter with cuda :D" << endl;
				// Copy the data to be used in the GPU helpers, just sensors in this case
				for ( int q = 0; q < NUM_SENSORS; q++ )
				{
					m_hSensorsZ[q] = vSensors[q]->z();
					m_hSensorsAng[q] = vSensors[q]->angle();
				}

				rb_pf_sensor_model_step( m_hParticles, NUM_PARTICLES,
										 m_hLines, m_hNumLines,
										 m_hSensorsZ, m_hSensorsAng, NUM_SENSORS );

				// Initialize normalizer
				float nrm = 0.0f;

				for ( int q = 0; q < NUM_PARTICLES; q++ )
				{
					m_particles[q].wz = m_hParticles[q].wz;
					nrm += m_particles[q].wz;

					m_particleWeights[q] = m_particles[q].wz;
					m_cdfParticleWeights[q] = 0.0f;
				}

				#else

				// Initialize normalizer
				float nrm = 0.0f;

				for ( int q = 0; q < NUM_PARTICLES; q++ )
				{
					// Compute importance weight *******************************************
					m_particles[q].wz = 1.0f;
					// Calculate the prob for each sensor applying sensor model give map
					for ( int p = 0; p < vSensors.size(); p++ )
					{
						m_particles[q].wz *= probSensorModel( m_particles[q], 
															  vSensors[p]->z(), 
															  vSensors[p]->angle(),
															  vMapWalls );
					}
					// update normalizer
					nrm += m_particles[q].wz;
					// *********************************************************************

					m_particleWeights[q] = m_particles[q].wz;
					m_cdfParticleWeights[q] = 0.0f;
				}


				#endif


				// normalize weights
				for ( int q = 0; q < NUM_PARTICLES; q++ )
				{
					m_particleWeights[q] /= nrm;
				}

				// Resampling ****************************

				// Calculate cdf
				m_cdfParticleWeights[0] = m_particleWeights[0];
				for ( int q = 2; q < NUM_PARTICLES; q++ )
				{
					m_cdfParticleWeights[q] += m_cdfParticleWeights[q - 1] + m_particleWeights[q];
					m_resampleIndxs[q] = 0;
				}

				float _threshold = RAND( 1.0f / NUM_PARTICLES );
				
				int i = 0;

				// systematic resampling
				for ( int q = 0; q < NUM_PARTICLES; q++ )
				{
					while( _threshold > m_cdfParticleWeights[i] )
					{
						i++;
					}

					if ( i >= NUM_PARTICLES )
					{
						// std::cout << "???" << std::endl;
						i = NUM_PARTICLES - 1;
					}

					m_resampleIndxs[q] = i;
					_threshold += ( 1.0f / NUM_PARTICLES );

				}

				// Regenerate the particles from the indxs obtained in the systematic resampling

				for ( int q = 0; q < NUM_PARTICLES; q++ )
				{
					m_particles[q].x = m_particles[m_resampleIndxs[q]].x;
					m_particles[q].y = m_particles[m_resampleIndxs[q]].y;
					m_particles[q].t = m_particles[m_resampleIndxs[q]].t;
				}

				// ***************************************

				// ****************************************************************************
			}

			for ( int q = 0; q < DRAW_PARTICLES; q++ )
			{
				engine::gl::LPrimitivesRenderer2D::instance->updatePoint( m_glIndxs[q], 
																		  m_particles[q].x, 
																		  m_particles[q].y );
			}

		}


		float LRobotLocalizer::probSensorModel( const LParticle& pParticle, 
												float zExp, float sensAngle,
												const vector<LLine>& vMapWalls )
		{
			float z = MAX_LEN;
			for ( int q = 0; q < vMapWalls.size(); q++ )
			{
				// Check with each line in the world

				float _pr_x = pParticle.x;
				float _pr_y = pParticle.y;

				float _p1_x = vMapWalls[q].p1.x;
				float _p1_y = vMapWalls[q].p1.y;

				float _dx = vMapWalls[q].p2.x - vMapWalls[q].p1.x;
				float _dy = vMapWalls[q].p2.y - vMapWalls[q].p1.y;
				float _dlen = sqrt( _dx * _dx + _dy * _dy );

				float _ul_x = _dx / _dlen;
				float _ul_y = _dy / _dlen;

				float _ur_x = cos( pParticle.t + sensAngle );
				float _ur_y = sin( pParticle.t + sensAngle );

				float _det = _ur_x * _ul_y - _ur_y * _ul_x;

				float _dx_rl = _pr_x - _p1_x;
				float _dy_rl = _pr_y - _p1_y;

				float _t = ( -_ur_y * _dx_rl + _ur_x * _dy_rl ) / _det;
				float _q = ( -_ul_y * _dx_rl + _ul_x * _dy_rl ) / _det;

				if ( _t > 0 && _t < _dlen )
				{
					if ( z > _q && _q > 0 )
					{
						z = _q;
					}
				}
			}

			return ( 1.0f / sqrt( 2 * PI * SIGMA_SENSOR * SIGMA_SENSOR ) ) *
						exp( -0.5 * ( z - zExp ) * ( z - zExp ) / ( SIGMA_SENSOR * SIGMA_SENSOR ) );
		}

		int LRobotLocalizer::calcNumParticlesInRange( float px, float py, float dRange )
		{
			int _count = 0;

			for ( int q = 0; q < NUM_PARTICLES; q++ )
			{
				float dx = m_particles[q].x - px;
				float dy = m_particles[q].y - py;

				if ( sqrt( dx * dx + dy * dy ) < dRange )
				{
					_count++;
				}
			}

			return _count;
		}

		void LRobotLocalizer::dumpInfo()
		{
			int _countClose = calcNumParticlesInRange( m_parent->getX(), m_parent->getY(), 20 );
			cout << "LRobotLocalizer::dumpInfo> inRange: " << _countClose << "/" << NUM_PARTICLES << endl;
		}

		void LRobotLocalizer::dumpParticles()
		{
			for ( int q = 0; q < DRAW_PARTICLES; q++ )
			{
				/* 
				cout << "x: " << m_particles[q].x << " - y: " << m_particles[q].y << endl;
				float dx = m_parent->getX() - m_particles[q].x;
				float dy = m_parent->getY() - m_particles[q].y;
				cout << "err: " << sqrt( dx * dx + dy * dy ) << endl;
				*/

				cout << "indx: " << m_resampleIndxs[q] << endl;
			}
		}

	}

}