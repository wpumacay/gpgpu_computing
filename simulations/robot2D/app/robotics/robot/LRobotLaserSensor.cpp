
#include "LRobotLaserSensor.h"
#include "../LRoboticsMap2D.h"
#include "../LAppRobotics2D.h"

#include <cmath>
#include <vector>

using namespace std;

namespace app
{

	namespace robotics2D
	{

			LRobotLaserSensor::LRobotLaserSensor( LRobot2D* parent, float angle )
			{
				m_parent = parent;
				m_angle = angle;
				m_len = MAX_LEN;

				float px = m_parent->getX();
				float py = m_parent->getY();
				float pTheta = m_parent->getTheta();

				float pxEnd = px + m_len * cos( pTheta + m_angle );
				float pyEnd = py + m_len * sin( pTheta + m_angle );

				// initialize primitive
				m_pIndx = engine::gl::LPrimitivesRenderer2D::instance->addLine( px, py, pxEnd, pyEnd,
																				1.0f, 1.0f, 1.0f );
			}

			void LRobotLaserSensor::update( float dt )
			{
				rayCast();

				float px = m_parent->getX();
				float py = m_parent->getY();
				float pTheta = m_parent->getTheta();

				float pxEnd = px + m_len * cos( pTheta + m_angle );
				float pyEnd = py + m_len * sin( pTheta + m_angle );

				engine::gl::LPrimitivesRenderer2D::instance->updateLine( m_pIndx, px, py, pxEnd, pyEnd );
			}

			void LRobotLaserSensor::rayCast()
			{
				vector<LLine> _lines = reinterpret_cast<LRoboticsWorld2D*>
											( LAppRobotics2D::instance->world() )->map()->lines();

				m_len = MAX_LEN;
				for ( int q = 0; q < _lines.size(); q++ )
				{
					// Check with each line in the world

					float _pr_x = m_parent->getX();
					float _pr_y = m_parent->getY();

					float _p1_x = _lines[q].p1.x;
					float _p1_y = _lines[q].p1.y;

					float _dx = _lines[q].p2.x - _lines[q].p1.x;
					float _dy = _lines[q].p2.y - _lines[q].p1.y;
					float _dlen = sqrt( _dx * _dx + _dy * _dy );

					float _ul_x = _dx / _dlen;
					float _ul_y = _dy / _dlen;

					float _ur_x = cos( m_parent->getTheta() + m_angle );
					float _ur_y = sin( m_parent->getTheta() + m_angle );

					float _det = _ur_x * _ul_y - _ur_y * _ul_x;

					float _dx_rl = _pr_x - _p1_x;
					float _dy_rl = _pr_y - _p1_y;

					float _t = ( -_ur_y * _dx_rl + _ur_x * _dy_rl ) / _det;
					float _q = ( -_ul_y * _dx_rl + _ul_x * _dy_rl ) / _det;

					if ( _t > 0 && _t < _dlen )
					{
						if ( m_len > _q && _q > 0 )
						{
							m_len = _q;
						}
					}
				}
			}

	}



}