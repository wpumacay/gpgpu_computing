
#pragma once

#include "../../gl/LApp.h"
#include "LParticlesWorld2D.h"


namespace app
{

	namespace robotics2D
	{

		class LAppParticles2D : public engine::gl::LApp
		{

			public :

			LAppParticles2D() : engine::gl::LApp()
			{

			}

			static void create();

			void createWorld() override;

		};



	}

}


