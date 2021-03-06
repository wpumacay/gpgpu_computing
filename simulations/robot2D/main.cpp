

#include "app/graphics/LAppGraphics2D.h"
#include "app/robotics/LAppRobotics2D.h"
#include "app/particles2D/LAppParticles2D.h"

using namespace app;

//#define TEST_GRAPHICS 1
#define TEST_ROBOTICS 1
//#define TEST_PARTICLES 1

int main()
{

#ifdef TEST_GRAPHICS

	graphics2D::LAppGraphics2D::create();

	graphics2D::LAppGraphics2D::instance->loop();		

	graphics2D::LAppGraphics2D::destroy();

#elif defined( TEST_ROBOTICS )

	robotics2D::LAppRobotics2D::create();

	robotics2D::LAppRobotics2D::instance->loop();		

	robotics2D::LAppRobotics2D::destroy();

#elif defined( TEST_PARTICLES )

	particles2D::LAppParticles2D::create();

	particles2D::LAppParticles2D::instance->loop();

	particles2D::LAppParticles2D::destroy();

#endif

	return 0;
}