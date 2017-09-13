
#pragma once

#include "LPrimitive.h"

#include <iostream>

using namespace std;

#define POINT_SIZE 0.005f

namespace engine
{

	namespace gl
	{

		class LPrimitivePoint : public LPrimitive
		{

			public :

			LPrimitivePoint() : LPrimitive()
			{
				this->type = primitive::PRIMITIVE_POINT;
			}

			LPrimitivePoint( float px, float py ) : LPrimitive()
			{
				this->type = primitive::PRIMITIVE_POINT;
				this->xy.x = px;
				this->xy.y = py;
			}

			void initGeometry() override
			{

				m_numVertices = 1;
				m_vertices = new GLfloat[ 3 * m_numVertices ];
				m_vertices[0] = 0.0f;
				m_vertices[1] = 0.0f;
				m_vertices[2] = 0.0f;

				programResIndx = LShaderManager::instance->createProgramAdv( "gl/core/shaders/primitives/gl_primitive_circle_vertex_shader.glsl",
																			 "gl/core/shaders/primitives/gl_primitive_circle_fragment_shader.glsl",
																		 	 "gl/core/shaders/primitives/gl_primitive_circle_geometry_shader.glsl" );

			}

			void drawGeometry( const LRenderInfo& rInfo, GLuint programId ) override
			{
				//cout << "rendering" << endl;
				GLuint u_cRadius = glGetUniformLocation( programId, "u_cRadius" );

				glUniform1f( u_cRadius, POINT_SIZE );

				glDrawArrays( GL_POINTS, 0, m_numVertices );
			}

		};


	}


}