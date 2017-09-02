

#pragma once

#include "../gl/LScene.h"


namespace app
{



	class LWorld2D
	{

		protected :

		engine::gl::LScene* m_scene;

		public :

		LWorld2D()
		{

			m_scene = new engine::gl::LScene();
		}

		~LWorld2D()
		{
			delete m_scene;
		}

		engine::gl::LScene* scene()
		{
			return m_scene;
		}


		virtual void update( float dt ) = 0;


	};







}