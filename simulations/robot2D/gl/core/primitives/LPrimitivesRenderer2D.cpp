
#include "LPrimitivesRenderer2D.h"
#include <iostream>

using namespace std;

namespace engine
{

	namespace gl
	{

		LPrimitivesRenderer2D* LPrimitivesRenderer2D::instance = NULL;

		LPrimitivesRenderer2D::LPrimitivesRenderer2D()
		{

		}

		LPrimitivesRenderer2D::~LPrimitivesRenderer2D()
		{
			for ( int q = 0; q < 5; q++ )
			{
				for ( int s = 0; s < m_primitivesPools[q].size(); s++ )
				{
					delete m_primitivesPools[q][s];
					m_primitivesPools[q][s] = NULL;
				}
			}
		}

		void LPrimitivesRenderer2D::create()
		{
			if ( engine::gl::LPrimitivesRenderer2D::instance != NULL )
			{
				delete engine::gl::LPrimitivesRenderer2D::instance;
			}

			engine::gl::LPrimitivesRenderer2D::instance = new engine::gl::LPrimitivesRenderer2D;
		}


		void LPrimitivesRenderer2D::render( const LRenderInfo& rInfo )
		{
			for ( int q = 0; q < 5; q++ )
			{
				for ( int s = 0; s < m_primitivesPools[q].size(); s++ )
				{
					m_primitivesPools[q][s]->render( rInfo );
				}
			}
		}

		void LPrimitivesRenderer2D::addPoint( float px, float py,
		                                      float r, float g, float b )
		{
			LPrimitivePoint* _point = new LPrimitivePoint();
			_point->init();
			_point->setColor( r, g, b, 1.0f );
			_point->xy.x = px;
			_point->xy.y = py;

			m_primitivesPools[primitive::PRIMITIVE_POINT].push_back( _point );
		}

		void LPrimitivesRenderer2D::addLine( float p1x, float p1y, float p2x, float p2y,
								             float r, float g, float b )
		{
			LPrimitiveLine* _line = new LPrimitiveLine( p1x, p1y, p2x, p2y );
			_line->init();
			_line->setColor( r, g, b, 1.0f );

			m_primitivesPools[primitive::PRIMITIVE_LINE].push_back( _line );
		}

		void LPrimitivesRenderer2D::addTriangle( float p1x, float p1y,
								                 float p2x, float p2y,
								                 float p3x, float p3y,
								                 float r, float g, float b )
		{

		}

		void LPrimitivesRenderer2D::addQuad( float p1x, float p1y,
								             float p2x, float p2y,
								             float p3x, float p3y,
								             float p4x, float p4y,
								             float r, float g, float b )
		{

		}

		void LPrimitivesRenderer2D::addCircle( float cx, float cy, float radius,
								               float r, float g, float b )
		{

		}

	}


}

