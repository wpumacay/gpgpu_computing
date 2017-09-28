
#pragma once

#include "../../gl/core/world/LWorld2D.h"
#include "LRoboticsMap2D.h"
#include "robot/LRobot2D.h"
#include "LCommonRobotics2D.h"

#include <iostream>
#include <string>

using namespace std;

#define CAM_SPEED_X 100.0f
#define CAM_SPEED_Y 100.0f

namespace app
{

    namespace robotics2D
    {

        class LRoboticsWorld2D : public engine::gl::LWorld2D
        {
        	private :

        	LRoboticsMap2D* m_map;

            LRobot2D* m_robot;

            public :

            LRoboticsWorld2D( float wWidth, float wHeight,
                              float appWidth, float appHeight,
                              float pix2world ) : engine::gl::LWorld2D( wWidth, wHeight,
                                                                        appWidth, appHeight,
                                                                        pix2world )
            {
                //m_robot = new LRobot2D( -17.5f, 762.5f );
                m_robot = new LRobot2D( 0.0f, 0.0f );
                m_robot->init();
                m_robot->setColor( 0.0f, 1.0f, 0.0f, 1.0f );

                //m_robot->setV( 50.0f );
                //m_robot->setW( 2.0f * PI / 5.0f );

                m_scene->addObject2D( m_robot );
            }

            ~LRoboticsWorld2D()
            {
                delete m_map;
                delete m_robot;
            }

            void setMap( LRoboticsMap2D* map )
            {
            	m_map = map;
            }

            LRoboticsMap2D* map()
            {
                return m_map;
            }

            void update( float dt )
            {
                m_camera->update( dt );

                // Get map info from Map helper
                vector<LLine> _mapWalls = m_map->lines();
                m_robot->update( dt, _mapWalls );
            }

            void onKeyDown( int pKey ) override
            {
                if ( pKey == GLFW_KEY_A ||
                     pKey == GLFW_KEY_W ||
                     pKey == GLFW_KEY_S ||
                     pKey == GLFW_KEY_D ||
                     pKey == GLFW_KEY_SPACE )
                {
                    m_robot->onKeyDown( pKey );
                }
            }

            void onKeyUp( int pKey ) override
            {
                if ( pKey == GLFW_KEY_A ||
                     pKey == GLFW_KEY_W ||
                     pKey == GLFW_KEY_S ||
                     pKey == GLFW_KEY_D ||
                     pKey == GLFW_KEY_SPACE )
                {
                    m_robot->onKeyUp( pKey );
                }
            }

        };

    }

}