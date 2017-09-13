
#pragma once

#include "../../gl/core/world/LWorld2D.h"
#include "../../gl/core/primitives/LPrimitivesRenderer2D.h"

#include <iostream>

using namespace std;
using namespace engine;

namespace app
{

    namespace graphics2D
   {

        class LGraphicsWorld2D : public engine::gl::LWorld2D
        {

            public :

            LGraphicsWorld2D( float wWidth, float wHeight,
                              float appWidth, float appHeight,
                              float pix2world ) : engine::gl::LWorld2D( wWidth, wHeight,
                                                                        appWidth, appHeight,
                                                                        pix2world )
            {
                // Create some graphics primitives
                gl::LPrimitivesRenderer2D::instance->addPoint( 0.0f, 0.0f );
                gl::LPrimitivesRenderer2D::instance->addPoint( 200.0f, 200.0f );
                //gl::LPrimitivesRenderer2D::instance->addLine( 0.0f, 0.0f, 200.0f, 200.0f );
                
                gl::LPrimitivesRenderer2D::instance->addPoint( 0.0f, 0.0f );
                gl::LPrimitivesRenderer2D::instance->addPoint( 200.0f, -200.0f );
                //gl::LPrimitivesRenderer2D::instance->addLine( 0.0f, 0.0f, 200.0f, -200.0f );

                gl::LPrimitivesRenderer2D::instance->addPoint( 0.0f, 0.0f );
                gl::LPrimitivesRenderer2D::instance->addPoint( -200.0f, -200.0f );
                //gl::LPrimitivesRenderer2D::instance->addLine( 0.0f, 0.0f, -200.0f, -200.0f );

                gl::LPrimitivesRenderer2D::instance->addPoint( 0.0f, 0.0f );
                gl::LPrimitivesRenderer2D::instance->addPoint( -200.0f, 200.0f );
                //gl::LPrimitivesRenderer2D::instance->addLine( 0.0f, 0.0f, -200.0f, 200.0f );

                gl::LPrimitivesRenderer2D::instance->addPoint( 400.0f, 0.0f );
                gl::LPrimitivesRenderer2D::instance->addPoint( 600.0f, 0.0f );
                

                gl::LPrimitivesRenderer2D::instance->addLine( -200.0f, 200.0f, 200.0f, 200.0f );
                //gl::LPrimitivesRenderer2D::instance->addLine( 200.0f, 200.0f, 200.0f, -200.0f );
                //gl::LPrimitivesRenderer2D::instance->addLine( 200.0f, -200.0f, -200.0f, -200.0f );
                //gl::LPrimitivesRenderer2D::instance->addLine( -200.0f, -200.0f, -200.0f, 200.0f );
            }

            void update( float dt )
            {
                m_camera->update( dt );
            }

        };

    }

}