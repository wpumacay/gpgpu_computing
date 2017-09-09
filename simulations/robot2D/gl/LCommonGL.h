
#pragma once


namespace engine
{

	namespace gl
	{

		struct LRenderInfo
		{
			int appWidth;
			int appHeight;
            int worldWidth;
            int worldHeight;
		};
	
		struct LPoint2D
		{
			double x;
			double y;

			LPoint2D()
			{
				x = 0;
				y = 0;
			}

			LPoint2D( double px, double py )
			{
				x = px;
				y = py;
			}
		};

	}

}