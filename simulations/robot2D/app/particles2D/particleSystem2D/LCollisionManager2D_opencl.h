

#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>

#include <CL/cl.hpp>

#include "cl/clUtils.h"
#include "../LCommonParticles2D.h"

using namespace std;

//#define USE_QUADTREE 1
//#define USE_GRID 	 1


namespace app
{


	namespace particles2D
	{


		class LCollisionManager2D_opencl
		{

			private :

			cl::Platform 	m_platform;
			cl::Device 		m_device;
			cl::Context 	m_context;

			cl::Program* 	m_program;

			public :

			LCollisionManager2D_opencl()
			{
			    vector<cl::Platform> v_platforms;
			    cl::Platform::get( &v_platforms );

			    // Choose the platform, just the first one for now
			    m_platform = v_platforms.front();

			    vector<cl::Device> v_devices;
			    m_platform.getDevices( CL_DEVICE_TYPE_ALL, &v_devices );
			    
			    // Choose the device
			    m_device = v_devices.front();

			    m_context = cl::Context( m_device );
			}

			~LCollisionManager2D_opencl()
			{

			}

			void _checkCollision_line_particle( LParticle& particle,
												const LLine& line )
			{
				
				if ( particle.vx == 0 && particle.vy == 0 )
				{
					return;
				}

				// first check if the point is in the inner side of the line
				float dx = particle.x - line.p1.x;
				float dy = particle.y - line.p1.y;
				float len = sqrt( dx * dx + dy * dy );
				float ux = dx / len;
				float uy = dy / len;

				float dot_v_normal = line.nx * ux + line.ny * uy;
				if ( dot_v_normal >= 0 )
				{
					// is in the inside region of the line, so no collision
					return;
				}

				// Check collision point
				float _ul_x = line.ux;
				float _ul_y = line.uy;

				float _up_x = particle.vx;
				float _up_y = particle.vy;
				float _vp = sqrt( _up_x * _up_x + _up_y * _up_y );
				_up_x = _up_x / _vp;
				_up_y = _up_y / _vp;

				float _det = _up_x * _ul_y - _up_y * _ul_x;

				float _dx_rl = particle.x - line.p1.x;
				float _dy_rl = particle.y - line.p1.y;

				float _t = ( -_up_y * _dx_rl + _up_x * _dy_rl ) / _det;
				float _q = ( -_ul_y * _dx_rl + _ul_x * _dy_rl ) / _det;

				if ( _t <= 0 || _t >= line.len || _q > 0 )
				{
					// If ray touches the line outside the segment or ...
					// if the ray is in the positive mov direction of the particle
					return;
				}
				// cout << "hit!!!" << endl;
				// cout << "_t: " << _t << " - _q: " << _q << endl;
				// Must have hit this wall, calculate back-return distance

				float _cos = _up_x * _ul_x + _up_y * _ul_y;
				float _sin = sqrt( 1 - _cos * _cos );
				float _back_ret_dist = abs( _q ) + ( particle.r / _sin );

				// return the particle to the place where it just hits the wall
				particle.x -= _up_x * _back_ret_dist;
				particle.y -= _up_y * _back_ret_dist;

				// Apply elastic hit
				float _vt = particle.vx * line.ux + particle.vy * line.uy;
				float _vn = particle.vx * line.nx + particle.vy * line.ny;
				_vn *= -1;
				// cout << "_vn: " << _vn << " - _vt: " << _vt << endl;
				// cout << "_nx " << line.nx << " - _ny: " << line.ny << endl;

				particle.vx = line.ux * _vt + line.nx * _vn;
				particle.vy = line.uy * _vt + line.ny * _vn;
			}

			void checkWorldBoundaryCollisions( float dt, 
										  	   vector<LParticle>& vParticles,
										  	   const vector<LLine>& vBoundaryWalls )
			{
				// This part is just in charge of ensuring the particles bounce
				// inside the given world limits

				for ( int q = 0; q < vParticles.size(); q++ )
				{
					// check with each boundary limit
					for ( int p = 0; p < vBoundaryWalls.size(); p++ )
					{
						_checkCollision_line_particle( vParticles[q], vBoundaryWalls[p] );
					}
				}

			}



			void checkWorldCollisions( float dt,
									   vector<LParticle>& vParticles
									   /* TODO */ )
			{
				// TODO
				// This part should be in charge of checking collisions of particles with general ...
				// rather complex world maps. Should pass a vector of these boundaries alongside ...
				// the vector of particles


			}

			void checkParticleCollisions( float dt, 
										  vector<LParticle>& vParticles )
			{
				// TODO
				// Use a quadtree to check collisions of just the necessary particles ...
				// , or maybe other another kind of group
				#ifdef USE_GRID

				gridhashCompact( vParticles );

				#elif defined( USE_QUADTREE )

				quadtreeCompact( vParticles );

				#endif



			}

			void gridhashCompact( vector<LParticle>& vParticles )
			{
				// TODO: Implement grid-hash collisions helper
			}

			void quadtreeCompact( vector<LParticle>& vParticles )
			{
				// TODO: Implement quadtree collisions helper
			}



		};






	}





}