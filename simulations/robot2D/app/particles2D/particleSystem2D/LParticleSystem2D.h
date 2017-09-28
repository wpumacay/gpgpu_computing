
#pragma once

#include "../LCommonParticles2D.h"
#include "../../../gl/core/primitives/LPrimitivesRenderer2D.h"

#include <vector>
#include <iostream>
#include <cmath>

#include "LCollisionManager2D.h"


using namespace std;

#define WALL_COLOR_R 0.0f
#define WALL_COLOR_G 0.0f
#define WALL_COLOR_B 1.0f

namespace app
{


	namespace particles2D
	{




		class LParticleSystem2D
		{

			private :

			vector<LParticle> m_particles;

			float m_worldSizeX;
			float m_worldSizeY;

			vector<LLine> m_walls;

			LCollisionManager2D* m_collisionManager;

			public :


			LParticleSystem2D( int pNumParticles,
							   float pSizeParticles,
							   float pWorldSizeX, float pWorldSizeY,
							   float pSpeedMin, float pSpeedMax )
			{

				m_worldSizeX = pWorldSizeX;
				m_worldSizeY = pWorldSizeY;

				// Initialize the world walls, should ...
				// make more sense in the world class, but just ...
				// put it here anyway

				// Simple rectangular-wall world
				// TODO: Maybe create configuration files for diff. worlds ...
				// and different properties like gravity, etc. in the particle system
				// Just port-copy the Map functionality in the robotics2d app

				// Create line which inner part points to the origin, so that ...  
				// the collision check returns them inside the boundaries
				LLine _wl( -0.5 * m_worldSizeX, -0.5 * m_worldSizeY,
						   -0.5 * m_worldSizeX,  0.5 * m_worldSizeY,
						   true, 0.0f, 0.0f );

				LLine _wt( -0.5 * m_worldSizeX, 0.5 * m_worldSizeY,
						    0.5 * m_worldSizeX, 0.5 * m_worldSizeY,
						    true, 0.0f, 0.0f );

				LLine _wr( 0.5 * m_worldSizeX,  0.5 * m_worldSizeY,
						   0.5 * m_worldSizeX, -0.5 * m_worldSizeY,
						   true, 0.0f, 0.0f );

				LLine _wb(  0.5 * m_worldSizeX, -0.5 * m_worldSizeY,
						   -0.5 * m_worldSizeX, -0.5 * m_worldSizeY,
						   true, 0.0f, 0.0f );

				m_walls.push_back( _wl );
				m_walls.push_back( _wt );
				m_walls.push_back( _wr );
				m_walls.push_back( _wb );

				// Initialize particles

				for ( int q = 0; q < pNumParticles; q++ )
				{
					LParticle _particle;
					_particle.x = RANDOM_SYM( -0.5 ) * m_worldSizeX;
					_particle.y = RANDOM_SYM( -0.5 ) * m_worldSizeY;

					_particle.r = pSizeParticles;

					float _v = pSpeedMin + ( pSpeedMax - pSpeedMin ) * RANDOM();
					float _t = 2 * PI * RANDOM();

					_particle.vx = _v * cos( _t );
					_particle.vy = _v * sin( _t );

					m_particles.push_back( _particle );
				}

				// create graphics primitives, just primitives for now

				for ( int q = 0; q < m_walls.size(); q++ )
				{
                	m_walls[q].glIndx = engine::gl::LPrimitivesRenderer2D::instance->addLine( m_walls[q].p1.x, m_walls[q].p1.y,
                														  					  m_walls[q].p2.x, m_walls[q].p2.y,
                														  					  WALL_COLOR_R, WALL_COLOR_G, WALL_COLOR_B );
				}

				for ( int q = 0; q < m_particles.size(); q++ )
				{
                	m_particles[q].glIndx = engine::gl::LPrimitivesRenderer2D::instance->addPoint( m_particles[q].x, m_particles[q].y );
				}

				m_collisionManager = new LCollisionManager2D();

			}


			~LParticleSystem2D()
			{
				delete m_collisionManager;
			}

			void addParticle( float px, float py )
			{
				LParticle _particle;
				_particle.x = px;
				_particle.y = py;

				_particle.r = PARTICLE_SIZE;

				float _v = PARTICLE_SPEED_MIN + ( PARTICLE_SPEED_MAX - PARTICLE_SPEED_MIN ) * RANDOM();
				float _t = RANDOM_SYM( 1.0f ) * PI;

				_particle.vx = _v * cos( _t );
				_particle.vy = _v * sin( _t );

				_particle.glIndx = engine::gl::LPrimitivesRenderer2D::instance->addPoint( _particle.x, _particle.y );

				m_particles.push_back( _particle );
			}

			void update( float dt )
			{
				for ( int q = 0; q < m_particles.size(); q++ )
				{
					m_particles[q].update( dt );
				}

				m_collisionManager->checkWorldBoundaryCollisions( dt,
																  m_particles,
																  m_walls );

				for ( int q = 0; q < m_particles.size(); q++ )
				{
					engine::gl::LPrimitivesRenderer2D::instance->updatePoint( m_particles[q].glIndx, 
																			  m_particles[q].x, 
																			  m_particles[q].y );
				}
			}
		};


	}





}









