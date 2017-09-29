
#pragma once

#include <vector>
#include "../LCommonRobotics2D.h"

#include "../../../gl/core/primitives/LPrimitivesRenderer2D.h"

using namespace std;

namespace app
{


	namespace robotics2D
	{


		class LLinePath
		{

			private :

			vector<LLine> m_lines;

			float m_startX;
			float m_startY;

			float m_endX;
			float m_endY;

			bool m_isClosedLoop;

			public :


			LLinePath( vector<float> xx, vector<float> yy,
					   bool _isClosedLoop = false )
			{
				m_startX = xx[0];
				m_startY = yy[0];

				m_isClosedLoop = _isClosedLoop;

				m_endX = ( _isClosedLoop ? m_startX : xx[xx.size() - 1] );
				m_endY = ( _isClosedLoop ? m_startY : yy[yy.size() - 1] );

				for ( int q = 0; q < xx.size() - 1; q++ )
				{
					LLine _line;
					_line.p1.x = xx[q];
					_line.p1.y = yy[q];
					_line.p2.x = xx[q + 1];
					_line.p2.y = yy[q + 1];

					m_lines.push_back( _line );
				}

				if ( _isClosedLoop )
				{
					LLine _line;
					_line.p1.x = xx[xx.size() - 1];
					_line.p1.y = yy[yy.size() - 1];
					_line.p2.x = xx[0];
					_line.p2.y = yy[0];					
					
					m_lines.push_back( _line );
				}

                for ( int q = 0; q < m_lines.size(); q++ )
                {
                	engine::gl::LPrimitivesRenderer2D::instance->addLine( m_lines[q].p1.x, m_lines[q].p1.y,
                														  m_lines[q].p2.x, m_lines[q].p2.y,
                														  0.0f, 0.5f, 0.5f );
                }
			}

			float startX()
			{
				return m_startX;
			}

			float startY()
			{
				return m_startY;
			}

			float endX()
			{
				return m_endX;
			}

			float endY()
			{
				return m_endY;
			}

			bool isClosedLoop()
			{
				return m_isClosedLoop;
			}

			void getDistance( float px, float py, float& d, float& pnx, float& pny )
			{
				d = 1000.0f;

				for ( int q = 0; q < m_lines.size(); q++ )
				{

					float _dx = m_lines[q].p2.x - m_lines[q].p1.x;
					float _dy = m_lines[q].p2.y - m_lines[q].p1.y;
					float _dlen = sqrt( _dx * _dx + _dy * _dy );

					float _ul_x = _dx / _dlen;
					float _ul_y = _dy / _dlen;

					float _un_x = -_ul_y;
					float _un_y = _ul_x;

					// float _det = _ur_x * _ul_y - _ur_y * _ul_x;
					float _det = -1;

					float _dx_lp = px - m_lines[q].p1.x;
					float _dy_lp = py - m_lines[q].p1.y;

					float _l = ( -_un_y * _dx_lp + _un_x * _dy_lp ) / _det;
					float _n = ( -_ul_y * _dx_lp + _ul_x * _dy_lp ) / _det;

					if ( _l > 0 && _l < _dlen )
					{
						if ( d > abs( _n ) )
						{
							d = abs( _n );
							pnx = m_lines[q].p1.x + _l * _ul_x;
							pny = m_lines[q].p1.y + _l * _ul_y;
						}
					}
				}
			}


		};

	}




}