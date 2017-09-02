

#include "gl/LApp.h"
#include "app/LWorld2D.h"


int main()
{
	double _tBef = 0;
	double _tNow = 0;
	double _tDelta = 0;

	engine::gl::LApp _app;
	_app.initialize();

	engine::gl::LScene* _stage = _app.stage();

	app::LParticleWorld2D* _world = new app::LParticleWorld2D();
	_stage->addChildScene( _world->scene() );

	while ( !glfwWindowShouldClose( _app.window() ) )
	{
		glfwPollEvents();

		_app->renderStep();

		_tNow = glfwGetTime();
		_tDelta = _tNow - _tBef;

		_world->update( _tDelta );
		
		_tBef = _tNow;

		glfwSwapBuffers( _app.window() )
	}

	return 0;
}