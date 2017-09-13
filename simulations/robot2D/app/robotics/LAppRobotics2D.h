
#pragma once

#include "../../gl/LApp.h"
#include "LRoboticsWorld2D.h"


namespace app
{

	namespace robotics2D
	{

		class LAppRobotics2D : public engine::gl::LApp
		{

			public :

			LAppRobotics2D() : engine::gl::LApp()
			{

			}

			static void create();

			void createWorld() override;

		};



	}

}


