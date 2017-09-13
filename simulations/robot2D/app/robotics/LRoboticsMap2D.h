
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "LCommonRobotics2D.h"
#include "../../gl/core/primitives/LPrimitivesRenderer2D.h"

using namespace std;

#define COLOR_MAP_R 0.0f
#define COLOR_MAP_G 0.0f
#define COLOR_MAP_B 1.0f

namespace app
{


	namespace robotics2D
	{

		class LRoboticsMap2D
		{
			private :

			vector<LPoint> m_points;
			vector<LLine> m_lines;

			float m_wWidth;
			float m_wHeight;

		    vector<string> split( const string &txt )
		    {
		        vector<string> _res;
		        
		        int pos = txt.find( ',' );
		        if ( pos == std::string::npos )
		        {
		            _res.push_back( txt );
		            return _res;
		        }

		        int initpos = 0;

		        while ( pos != std::string::npos )
		        {
		            _res.push_back( txt.substr( initpos, pos - initpos + 1 ) );
		            initpos = pos + 1;

		            pos = txt.find( ',', initpos );
		        }

		        _res.push_back( txt.substr( initpos, std::min( pos, (int) txt.size() ) - initpos + 1 ) );
		        
		        return _res;
		    }

			void parseMap( const string& mapFile )
			{
				ifstream _f_handle( mapFile.c_str() );
			
				if ( !_f_handle.is_open() )
				{
					throw "LRoboticsMap2D::parseMap> couldnt open map file";
				}

				string _line;
				getline( _f_handle, _line );

				vector<string> _wh = split( _line );
				float _wWidth  = stof( _wh[0] );
				float _wHeight = stof( _wh[1] );

				getline( _f_handle, _line );
				int _nPointsOut = stoi( _line );

				vector<float> _w_px_out;
				vector<float> _w_py_out;
				for ( int q = 0; q < _nPointsOut; q++ )
				{
					getline( _f_handle, _line );

					vector<string> _point = split( _line );

					_w_px_out.push_back( stof( _point[0] ) );
					_w_py_out.push_back( stof( _point[1] ) );
				}

				getline( _f_handle, _line );
				int _nPointsIn = stoi( _line );

				vector<float> _w_px_in;
				vector<float> _w_py_in;
				for ( int q = 0; q < _nPointsIn; q++ )
				{
					getline( _f_handle, _line );

					vector<string> _point = split( _line );

					_w_px_in.push_back( stof( _point[0] ) );
					_w_py_in.push_back( stof( _point[1] ) );
				}

				// create the batch of points
				for ( int q = 0; q < _w_px_out.size(); q++ )
				{
					LPoint _p;
					_p.x = _w_px_out[q] - _wWidth * 0.5;
					_p.y = -_w_py_out[q] + _wHeight * 0.5;

					m_points.push_back( _p );
				}
				for ( int q = 0; q < _w_px_in.size(); q++ )
				{
					LPoint _p;
					_p.x = _w_px_in[q] - _wWidth * 0.5;
					_p.y = -_w_py_in[q] + _wHeight * 0.5;

					m_points.push_back( _p );
				}

				// create the lines
				for ( int q = 0; q < _w_px_in.size(); q++ )
				{
					LLine _l;
					_l.p1.x = _w_px_in[q] - _wWidth * 0.5;
					_l.p1.y = -_w_py_in[q] + _wHeight * 0.5;
					_l.p2.x = _w_px_in[(q + 1) % _w_px_in.size()] - _wWidth * 0.5;
					_l.p2.y = -_w_py_in[(q + 1) % _w_py_in.size()] + _wHeight * 0.5;

					m_lines.push_back( _l );
				}
				for ( int q = 0; q < _w_px_out.size(); q++ )
				{
					LLine _l;
					_l.p1.x = _w_px_out[q] - _wWidth * 0.5;
					_l.p1.y = -_w_py_out[q] + _wHeight * 0.5;
					_l.p2.x = _w_px_out[(q + 1) % _w_px_out.size()] - _wWidth * 0.5;
					_l.p2.y = -_w_py_out[(q + 1) % _w_py_out.size()] + _wHeight * 0.5;

					m_lines.push_back( _l );
				}

				m_wWidth  = _wWidth;
				m_wHeight = _wHeight;
			}

			void initMapFromPrimitives()
			{
				/*
                for ( int q = 0; q < m_points.size(); q++ )
                {
                	engine::gl::LPrimitivesRenderer2D::instance->addPoint( m_points[q].x, m_points[q].y );
                	// cout << "px: " << m_points[q].x << endl;
                	// cout << "py: " << m_points[q].y << endl;
                }
				*/
                for ( int q = 0; q < m_lines.size(); q++ )
                {
                	engine::gl::LPrimitivesRenderer2D::instance->addLine( m_lines[q].p1.x, m_lines[q].p1.y,
                														  m_lines[q].p2.x, m_lines[q].p2.y,
                														  COLOR_MAP_R, COLOR_MAP_G, COLOR_MAP_B );
                }
			}

			public :


			LRoboticsMap2D( const string& mapFile )
			{
				parseMap( mapFile );
				initMapFromPrimitives();
			}

			~LRoboticsMap2D()
			{
				m_points.clear();
				m_lines.clear();
			}

			vector<LLine>& lines()
			{
				return m_lines;
			}

			float worldWidth()
			{
				return m_wWidth;
			}

			float worldHeight()
			{
				return m_wHeight;
			}

		};




	}



}