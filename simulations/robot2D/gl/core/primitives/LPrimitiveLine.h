
#pragma once

#include "LPrimitive.h"

#include <iostream>
#include <cmath>

using namespace std;

namespace engine
{

	namespace gl
	{

		class LPrimitiveLine : public LPrimitive
		{

			float dx_w;
			float dy_w;

			public :

			LPrimitiveLine( float p1x, float p1y, float p2x, float p2y ) : LPrimitive()
			{
				this->type = primitive::PRIMITIVE_LINE;
				this->xy.x = p1x;
				this->xy.y = p1y;

				this->dx_w = p2x - p1x;
				this->dy_w = p2y - p1y;
			}

			void updatePoints( float p1x, float p1y, float p2x, float p2y )
			{
				this->xy.x = p1x;
				this->xy.y = p1y;

				this->dx_w = p2x - p1x;
				this->dy_w = p2y - p1y;
			}

			void initGeometry() override
			{

				m_numVertices = 1;
				m_vertices = new GLfloat[ 3 * m_numVertices ];
				m_vertices[0] = 0.0f;
				m_vertices[1] = 0.0f;
				m_vertices[2] = 0.0f;

				programResIndx = LShaderManager::instance->createProgramAdv( "gl/core/shaders/primitives/gl_primitive_line_vertex_shader.glsl",
																		  	 "gl/core/shaders/primitives/gl_primitive_line_fragment_shader.glsl",
																		  	 "gl/core/shaders/primitives/gl_primitive_line_geometry_shader.glsl" );

			}

			void drawGeometry( const LRenderInfo& rInfo, GLuint programId ) override
			{
				GLuint u_dx = glGetUniformLocation( programId, "u_dx" );
				GLuint u_dy = glGetUniformLocation( programId, "u_dy" );

				float dx_clip = ( 2. * rInfo.cameraZoom ) * ( ( this->dx_w + rInfo.cameraX * 0.0 ) / rInfo.pix2world ) / rInfo.appWidth;
				float dy_clip = ( 2. * rInfo.cameraZoom ) * ( ( this->dy_w + rInfo.cameraY * 0.0 ) / rInfo.pix2world ) / rInfo.appHeight;

				//cout << "dx: " << this->dx_w << endl;
				//cout << "dy: " << this->dy_w << endl;
				//cout << "dxClip: " << dx_clip << endl;
				//cout << "dyClip: " << dy_clip << endl;

				glUniform1f( u_dx, dx_clip );
				glUniform1f( u_dy, dy_clip );

				glDrawArrays( GL_POINTS, 0, m_numVertices );
			}

		};


	}


}