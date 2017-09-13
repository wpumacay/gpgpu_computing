
#pragma once

#include "../../gl/LApp.h"
#include "LGraphicsWorld2D.h"


namespace app
{

	namespace graphics2D
	{

		class LAppGraphics2D : public engine::gl::LApp
		{

			public :

			LAppGraphics2D() : engine::gl::LApp()
			{

			}

			static void create();

			void createWorld() override;

		};



	}

}


