
#include "LRobot2D.h"
#include <GLFW/glfw3.h>

#include <iostream>
#include <algorithm>

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

				kp = 0.2f;
				ki = 0.05f;
				kd = 0.01f;

				err_now = 0.0f;
				err_bef = 0.0f;

				m_lastErrD = 1000.0f;
				m_lastErrX = m_x;
				m_lastErrY = m_y;

				ep = 0;
				ei = 0;
				ed = 0;

                m_errLine.glIndx = engine::gl::LPrimitivesRenderer2D::instance->addLine( m_errLine.p1.x, m_errLine.p1.y,
                													  					 m_errLine.p2.x, m_errLine.p2.y,
                													  					 1.0f, 0.0f, 0.0f );
			}

			void LRobot2D::reset( float x, float y, float theta )
			{
				m_x = x;
				m_y = y;
				m_theta = theta;
				m_localizer->reset();
			}

			void LRobot2D::update( float dt, vector<LLine> vMapWalls, LLinePath* pPath )
			{
				float _d, _pnx, _pny;
				pPath->getDistance( m_x, m_y, _d, _pnx, _pny );
				if ( abs( _d ) > 200.0f )
				{
					_d = m_lastErrD;
					_pnx = m_lastErrX;
					_pny = m_lastErrY;
				}
				else
				{
					m_lastErrD = _d;
					m_lastErrX = _pnx;
					m_lastErrY = _pny;
				}
				engine::gl::LPrimitivesRenderer2D::instance->updateLine( m_errLine.glIndx, m_x, m_y, _pnx, _pny );


				if ( !m_isInAutonomousMode )
				{
					m_v = m_manualControls[R_KEY_W] * ( R_MANUAL_V ) + 
						  m_manualControls[R_KEY_S] * ( -R_MANUAL_V );
					m_w = m_manualControls[R_KEY_A] * ( R_MANUAL_W )+
						  m_manualControls[R_KEY_D] * ( -R_MANUAL_W );
				}
				#ifdef ALLOW_AUTO_MODE
				else
				{
					m_v = R_MANUAL_V;

					// First attempt of a controller
					float _dnx = _pnx - m_x;
					float _dny = _pny - m_y;
					float _dlen = sqrt( _dnx * _dnx + _dny * _dny );
					if ( _dlen < 0.1f )
					{
						_dnx = 0.0f;
						_dny = 0.0f;
						_dlen = 1.0f;
					}

					float _unx = _dnx / _dlen;
					float _uny = _dny / _dlen;

					float _ulx = -_uny;
					float _uly = _unx;

					float _uvx = cos( m_theta );
					float _uvy = sin( m_theta );
					// Align in the correct direction
					float _dot_l_on_v = _ulx * _uvx + _uly * _uvy;
					if ( _dot_l_on_v < 0 )
					{
						_ulx *= -1;
						_uly *= -1;
					}

					err_now = _d;
					ep = err_now;
					ed = err_now - err_bef;
					ei += err_now;

					float _un = kp * ep + ki * ei + kd * ed;

					err_bef = err_now;

					float _dirX = 1000.0f * _ulx + _un * _unx;
					float _dirY = 1000.0f * _uly + _un * _uny;
					float _dirLen = 1000.0f;//sqrt( _dirX * _dirX + _dirY * _dirY );
					float _udirX = _dirX / _dirLen;
					float _udirY = _dirY / _dirLen;

					float _errTheta = _udirX * _uvy - _udirY * _uvx;

					m_w = - 10.f * _errTheta;

				}
				#endif
				
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