

#pragma once

#include <vector>

#include "../LCommonParticles2D.h"

using namespace std;

//#define USE_QUADTREE 1
//#define USE_GRID 	 1


namespace app
{


	namespace particles2D
	{


		class LCollisionManager
		{

			public :

			LCollisionManager()
			{

			}

			~LCollisionManager()
			{

			}

			void checkWorldBoundaryCollisions( float dt, 
										  	   vector<LParticle>& vParticles,
										  	   float xMin, float xMax, float yMin, float yMax )
			{
				// This part is just in charge of ensuring the particles bounce
				// inside the given world limits

				for ( int q = 0; q < vParticles.size(); q++ )
				{
					// Check left limit
					
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