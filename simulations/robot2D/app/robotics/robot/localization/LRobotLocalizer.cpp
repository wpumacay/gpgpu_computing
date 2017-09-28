
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
		}

		LRobotLocalizer::~LRobotLocalizer()
		{
			
		}

		void LRobotLocalizer::update( float dt, vector<LLine> vMapWalls )
		{

			for ( int q = 0; q < NUM_PARTICLES; q++ )
			{
				// update each particle based on the motion model
				m_particles[q].update( dt, m_parent->getV(), m_parent->getW() );
			}

			if ( m_useFilter )
			{
				// particle filter algorithm **************************************************
				vector<LRobotLaserSensor*> vSensors = m_parent->sensors();

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
					m_resampleIndxs[NUM_PARTICLES] = 0;
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

					m_resampleIndxs[q] = i;
					_threshold += ( 1.0f / NUM_PARTICLES );

					if ( i >= NUM_PARTICLES )
					{
						// std::cout << "???" << std::endl;
						i = NUM_PARTICLES - 1;
					}
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

		void LRobotLocalizer::dumpParticles()
		{
			for ( int q = 0; q < DRAW_PARTICLES; q++ )
			{
				cout << "x: " << m_particles[q].x << " - y: " << m_particles[q].y << endl;
				float dx = m_parent->getX() - m_particles[q].x;
				float dy = m_parent->getY() - m_particles[q].y;
				cout << "err: " << sqrt( dx * dx + dy * dy ) << endl;
			}
		}

	}

}