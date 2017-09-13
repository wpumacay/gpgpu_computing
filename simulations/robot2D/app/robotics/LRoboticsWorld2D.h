
#pragma once

#include "../../gl/core/world/LWorld2D.h"
#include "LRoboticsMap2D.h"

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


            public :

            LRoboticsWorld2D( float wWidth, float wHeight,
                              float appWidth, float appHeight,
                              float pix2world ) : engine::gl::LWorld2D( wWidth, wHeight,
                                                                        appWidth, appHeight,
                                                                        pix2world )
            {
                
            }

            void setMap( LRoboticsMap2D* map )
            {
            	m_map = map;
            }

            void update( float dt )
            {
                m_camera->update( dt );
            }

            void onKeyDown( int pKey ) override
            {
            	if ( pKey == GLFW_KEY_W )
            	{
            		m_camera->setVy( -CAM_SPEED_Y );
            	}
            	else if ( pKey == GLFW_KEY_S )
            	{
            		m_camera->setVy( CAM_SPEED_Y );
            	}
            	else if ( pKey == GLFW_KEY_D )
            	{
            		m_camera->setVx( -CAM_SPEED_X );
            	}
            	else if ( pKey == GLFW_KEY_A )
            	{
            		m_camera->setVx( CAM_SPEED_X );
            	}
            }

            void onKeyUp( int pKey ) override
            {
            	if ( pKey == GLFW_KEY_W )
            	{
            		m_camera->setVy( 0.0f );
            	}
            	else if ( pKey == GLFW_KEY_S )
            	{
            		m_camera->setVy( 0.0f );
            	}
            	else if ( pKey == GLFW_KEY_D )
            	{
            		m_camera->setVx( 0.0f );
            	}
            	else if ( pKey == GLFW_KEY_A )
            	{
            		m_camera->setVx( 0.0f );
            	}
            }

        };

    }

}