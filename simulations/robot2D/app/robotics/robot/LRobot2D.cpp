
#include "LRobot2D.h"

namespace app
{

	namespace robotics2D
	{

			LRobot2D::LRobot2D( float x, float y ) : engine::gl::LBaseObject2D()
			{
				m_x = x;
				m_y = y;
				m_theta = 0.0f;
				this->xy.x = x;
				this->xy.y = y;

				m_v = 0.0f;
				m_w = 0.0f;

				for ( int q = 0; q < NUM_SENSORS; q++ )
				{
					LRobotLaserSensor* _sensor = new LRobotLaserSensor( this, 
																		( q - NUM_SENSORS / 2. + 0.5 ) * PI / ( NUM_SENSORS - 1 ) );

					m_sensors.push_back( _sensor );
				}

				m_localizer = new LRobotLocalizer( this );
			}

			void LRobot2D::update( float dt )
			{
				
				if ( abs( m_w ) < 0.0001f )
				{
					m_x += m_v * dt * cos( m_theta );
					m_y += m_v * dt * sin( m_theta );
				}
				else
				{
					m_x += ( m_v / m_w ) * ( sin( m_theta + m_w * dt ) - sin( m_theta ) );
					m_y += ( m_v / m_w ) * ( -cos( m_theta + m_w * dt ) + cos( m_theta ) );
					m_theta += m_w * dt;
				}


				this->xy.x = m_x;
				this->xy.y = m_y;
				//this->rotation = m_theta;

				for ( int q = 0; q < m_sensors.size(); q++ )
				{
					m_sensors[q]->update( dt );
				}

				m_localizer->update( dt );
			}

			void LRobot2D::setV( float v )
			{
				m_v = v;
			}

			void LRobot2D::setW( float w )
			{
				m_w = w;
			}

			void LRobot2D::setX( float x )
			{
				m_x = x;
			}

			void LRobot2D::setY( float y )
			{
				m_y = y;
			}

			void LRobot2D::setTheta( float theta )
			{
				m_theta = theta;
			}

			float LRobot2D::getX() 
			{ 
				return m_x; 
			}

			float LRobot2D::getY() 
			{ 
				return m_y; 
			}

			float LRobot2D::getTheta() 
			{ 
				return m_theta; 
			}

			float LRobot2D::getV() 
			{ 
				return m_v; 
			}

			float LRobot2D::getW() 
			{ 
				return m_w; 
			}

	}

}