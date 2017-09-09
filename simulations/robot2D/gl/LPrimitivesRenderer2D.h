
#pragma once

#include <GL/glew.h>
#include <vector>
#include "LCommonGL.h"

#define DEFAULT_COLOR_R 0.0f
#define DEFAULT_COLOR_G 0.1f
#define DEFAULT_COLOR_B 0.0f

namespace engine
{

    namespace gl
    {

        class LPrimitivesRenderer2D
        {

            public :


            static void drawPoint( const LRenderInfo& rInfo,
                                   float px, float py,
                                   float r = DEFAULT_COLOR_R, 
                                   float g = DEFAULT_COLOR_G, 
                                   float b = DEFAULT_COLOR_B );

            static void drawLine( const LRenderInfo& rInfo,
                                  float p1x, float p1y, float p2x, float p2y
                                  float r = DEFAULT_COLOR_R, 
                                  float g = DEFAULT_COLOR_G, 
                                  float b = DEFAULT_COLOR_B );

            static void drawTriangle( const LRenderInfo& rInfo,
                                      float p1x, float p1y,
                                      float p2x, float p2y,
                                      float p3x, float p3y,
                                      float r = DEFAULT_COLOR_R, 
                                      float g = DEFAULT_COLOR_G, 
                                      float b = DEFAULT_COLOR_B );

            static void drawQuad( const LRenderInfo& rInfo,
                                  float p1x, float p1y,
                                  float p2x, float p2y,
                                  float p3x, float p3y,
                                  float p4x, float p4y,
                                  float r = DEFAULT_COLOR_R, 
                                  float g = DEFAULT_COLOR_G, 
                                  float b = DEFAULT_COLOR_B );

            static void drawCircle( const LRenderInfo& rInfo,
                                    float cx, float cy, float radius,
                                    float r = DEFAULT_COLOR_R, 
                                    float g = DEFAULT_COLOR_G, 
                                    float b = DEFAULT_COLOR_B );

            static void drawPointSwarm( const LRenderInfo& rInfo,
                                        float* px, float* py,
                                        float r = DEFAULT_COLOR_R, 
                                        float g = DEFAULT_COLOR_G, 
                                        float b = DEFAULT_COLOR_B );

            static void drawCircleSwarm( const LRenderInfo& rInfo,
                                         float* cx, float* cy, float* radius,
                                         int nCircles,
                                         float r = DEFAULT_COLOR_R, 
                                         float g = DEFAULT_COLOR_G, 
                                         float b = DEFAULT_COLOR_B );

            static void drawLineSwarm( const LRenderInfo& rInfo,
                                       float* p1x, float* p1y,
                                       float* p2x, float* p2y,
                                       int nLines,
                                       float r = DEFAULT_COLOR_R, 
                                       float g = DEFAULT_COLOR_G, 
                                       float b = DEFAULT_COLOR_B );
        };

    }


}

void engine::gl::LPrimitivesRenderer2D::drawPoint( const LRenderInfo& rInfo,
                                                   float px, float py,
                                                   float r = DEFAULT_COLOR_R, 
                                                   float g = DEFAULT_COLOR_G, 
                                                   float b = DEFAULT_COLOR_B )
{


    GLfloat* _vertex = new GLfloat[3];
    _vertex[0] = px;
    _vertex[1] = py;
    _vertex[2] = pz;

    




    delete[] _vertex;

}