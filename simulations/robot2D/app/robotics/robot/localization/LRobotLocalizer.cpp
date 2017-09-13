
#include "LRobotLocalizer.h"
#include "../../../../gl/core/primitives/LPrimitivesRenderer2D.h"

namespace app
{

	namespace robotics2D
	{


		LRobotLocalizer::LRobotLocalizer( LRobot2D* parent )
		{
			m_parent = parent;

			for ( int q = 0; q < NUM_PARTICLES; q++ )
			{
				m_particles.push_back( new LParticle( m_parent->getX(),
													  m_parent->getY(),
													  m_parent->getTheta() ) );
			}

			for ( int q = 0; q < m_particles.size(); q++ )
			{
				m_indxs.push_back( engine::gl::LPrimitivesRenderer2D::instance->addPoint( m_particles[q]->x, 
																						  m_particles[q]->y,
																						  1.0f, 1.0f, 1.0f ) );
			}
		}

		LRobotLocalizer::~LRobotLocalizer()
		{
			for ( int  q = 0; q < m_particles.size(); q++ )
			{
				delete m_particles[q];
			}
			m_particles.clear();
		}

		void LRobotLocalizer::update( float dt )
		{
			for ( int q = 0; q < m_particles.size(); q++ )
			{
				m_particles[q]->update( dt, m_parent->getV(), m_parent->getW() );
				engine::gl::LPrimitivesRenderer2D::instance->updatePoint( m_indxs[q], 
																		  m_particles[q]->x, 
																		  m_particles[q]->y );
			}
		}


	}

}