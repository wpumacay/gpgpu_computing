

#include "LAppRobotics2D.h"
#include "LRoboticsMap2D.h"

#include <iostream>
#include <string>

using namespace std;

namespace app
{

	namespace robotics2D
	{

		void LAppRobotics2D::create()
		{
			cout << "creating robotics2D app" << endl;

			if ( LAppRobotics2D::instance != NULL )
			{
				delete LAppRobotics2D::instance;
			}

			LAppRobotics2D::instance = new LAppRobotics2D();
			LAppRobotics2D::instance->initialize();
		}

		void LAppRobotics2D::createWorld()
		{
			cout << "creating graphics-world2d" << endl;

			LRoboticsMap2D* _r_map = new LRoboticsMap2D( string( "resources/robo_map_1.txt" ) );

			m_world = new LRoboticsWorld2D( _r_map->worldWidth(), _r_map->worldHeight(), 
                                    		APP_WIDTH, APP_HEIGHT, 
                                    		1.0f );
			reinterpret_cast<LRoboticsWorld2D*>( m_world )->setMap( _r_map );
			m_stage->addChildScene( m_world->scene() );
		}

	}

}