
#include "LRobot2D.h"
#include <GLFW/glfw3.h>

#include <iostream>

using namespace std;

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

				m_localizer->setFilterState( 0 );

				m_useLocalizerFilter = 0;

				m_isInAutonomousMode = true;
				m_manualControls[0] = 0;
				m_manualControls[1] = 0;
				m_manualControls[2] = 0;
				m_manualControls[3] = 0;
			}

			void LRobot2D::update( float dt, vector<LLine> vMapWalls )
			{

				if ( !m_isInAutonomousMode )
				{
					m_v = m_manualControls[R_KEY_W] * ( R_MANUAL_V ) + 
						  m_manualControls[R_KEY_S] * ( -R_MANUAL_V );
					m_w = m_manualControls[R_KEY_A] * ( R_MANUAL_W )+
						  m_manualControls[R_KEY_D] * ( -R_MANUAL_W );
				}
				
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

				m_localizer->setFilterState( m_useLocalizerFilter );

				m_localizer->update( dt, vMapWalls );
			}

			void LRobot2D::onMapLoaded( vector<LLine> wallLines )
			{
				m_localizer->onMapLoaded( wallLines );
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

			vector<LRobotLaserSensor*> LRobot2D::sensors()
			{
				return m_sensors;
			}

			void LRobot2D::onKeyDown( int pKey )
			{

				if ( pKey == GLFW_KEY_W )
				{
					m_manualControls[R_KEY_W] = 1;
				}
				else if ( pKey == GLFW_KEY_A )
				{
					m_manualControls[R_KEY_A] = 1;
				}
				else if ( pKey == GLFW_KEY_S )
				{
					m_manualControls[R_KEY_S] = 1;
				}
				else if ( pKey == GLFW_KEY_D )
				{
					m_manualControls[R_KEY_D] = 1;
				}

				m_isInAutonomousMode = true;
				for ( int q = 0; q < 4; q++ )
				{
					if ( m_manualControls[q] != 0 )
					{
						m_isInAutonomousMode = false;
						break;
					}
				}

				if ( pKey == GLFW_KEY_SPACE )
				{
					toogleFilter();
				}

			}

			void LRobot2D::onKeyUp( int pKey )
			{

				if ( pKey == GLFW_KEY_W )
				{
					m_manualControls[R_KEY_W] = 0;
				}
				else if ( pKey == GLFW_KEY_A )
				{
					m_manualControls[R_KEY_A] = 0;
				}
				else if ( pKey == GLFW_KEY_S )
				{
					m_manualControls[R_KEY_S] = 0;
				}
				else if ( pKey == GLFW_KEY_D )
				{
					m_manualControls[R_KEY_D] = 0;
				}



				m_isInAutonomousMode = true;
				for ( int q = 0; q < 4; q++ )
				{
					if ( m_manualControls[q] != 0 )
					{
						m_isInAutonomousMode = false;
						break;
					}
				}

				if ( m_isInAutonomousMode )
				{
					m_v = 0;
					m_w = 0;
				}
			}

			void LRobot2D::toogleFilter()
			{
				m_useLocalizerFilter = 1 - m_useLocalizerFilter;
				if ( m_useLocalizerFilter == 0 )
				{
					m_localizer->dumpParticles();
					m_localizer->dumpInfo();
				}
				// m_localizer->useFilter = !m_localizer->useFilter;
				// cout << "useFilter: " << ( m_localizer->useFilter ? "true" : "false" ) << endl;
			}
	}

}