

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

using namespace std;

namespace img
{



	struct LImageRGB
	{
		int w;
		int h;
		u8* buffer;

		LImageRGB( int width, int height )
		{
			w = width;
			h = height;

			buffer = new u8[width * height * 4];
                        for ( int q = 0; q < w; q++ )
                        {
                            for ( int p = 0; p < h; p++ )
                            {
                                int _off = q + p * w;
                                buffer[_off * 4 + 0] = 0;
                                buffer[_off * 4 + 1] = 0;
                                buffer[_off * 4 + 2] = 0;
                                buffer[_off * 4 + 3] = 255;
                            }
                        }
		}

		~LImageRGB()
		{
			delete[] buffer;
		}

		void saveImage( string imgFileName )
		{
			cv::Mat _res( h, w, CV_8UC4, ( void* ) buffer );
			cv::imwrite( imgFileName.c_str(), _res );
		}

	};





}
