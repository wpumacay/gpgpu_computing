

#include "LAppGraphics2D.h"

#include <iostream>

using namespace std;

namespace app
{

	namespace graphics2D
	{

		void LAppGraphics2D::create()
		{
			cout << "creating graphics2D app" << endl;

			if ( LAppGraphics2D::instance != NULL )
			{
				delete LAppGraphics2D::instance;
			}

			LAppGraphics2D::instance = new LAppGraphics2D();
			LAppGraphics2D::instance->initialize();
		}

		void LAppGraphics2D::createWorld()
		{
			cout << "creating graphics-world2d" << endl;

			m_world = new LGraphicsWorld2D( 4000.0f, 2000.0f, 
                                    		APP_WIDTH, APP_HEIGHT, 
                                    		1.0f );
			m_stage->addChildScene( m_world->scene() );
		}

	}

}