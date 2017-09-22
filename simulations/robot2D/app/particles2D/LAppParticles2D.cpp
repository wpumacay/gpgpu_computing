

#include "LAppParticles2D.h"

#include <iostream>
#include <string>

using namespace std;

namespace app
{

	namespace robotics2D
	{

		void LAppParticles2D::create()
		{
			cout << "creating particles2D app" << endl;

			if ( LAppParticles2D::instance != NULL )
			{
				delete LAppParticles2D::instance;
			}

			LAppParticles2D::instance = new LAppParticles2D();
			LAppParticles2D::instance->initialize();
		}

		void LAppParticles2D::createWorld()
		{
			cout << "creating particles2D's world" << endl;

			m_world = new LParticlesWorld2D( PARTICLE_SIZE_X, PARTICLE_SIZE_Y, 
                                    		 APP_WIDTH, APP_HEIGHT, 
                                    		 1.0f );
			
			m_stage->addChildScene( m_world->scene() );
		}

	}

}