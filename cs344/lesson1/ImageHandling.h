
#pragma once

#include "Common.h"


#include <iostream>
#include <string>

#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
#else
    #include <CL/cl.hpp>
#endif


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>



namespace IH
{


    void preProcess( uchar4 **h_inputImage, u8 **h_greyImage,
                     uchar4 **d_rgbaImage, u8 **d_greyImage,
                     size_t &nRows, size_t &nCols,
                     const std::string &filename )
    {
        cv::Mat *_imgRGBA = new cv::Mat();
        cv::Mat *_imgGrey = new cv::Mat();

        cv::Mat _img;
        _img = cv::imread( filename.c_str(), CV_LOAD_IMAGE_COLOR );
        if ( _img.empty() )
        {
            std::cout << "Couldnt open file" << std::endl;
            return;
        }
        
        cv::cvtColor( _img, *_imgRGBA, CV_BGR2RGBA );
        _imgGrey->create( _img.rows, _img.cols, CV_8UC1 );

        nRows = _img.rows;
        nCols = _img.cols;

        *h_inputImage = ( uchar4 * ) _imgRGBA->ptr<u8>( 0 );
        *h_greyImage = _imgGrey->ptr<u8>( 0 );

        size_t _numPixels = _imgRGBA->rows * _imgRGBA->cols;
        
        cudaMalloc( d_rgbaImage, sizeof( uchar4 ) * _numPixels );
        cudaMalloc( d_greyImage, sizeof( u8 ) * _numPixels );
        cudaMemset( *d_greyImage, 0, sizeof( u8 ) * _numPixels );

        cudaMemcpy( *d_rgbaImage, *h_inputImage, sizeof( uchar4 ) * _numPixels, cudaMemcpyHostToDevice );
    }
                     

    void postProcess( size_t numRows, size_t numCols, u8* data )
    {
        cv::Mat _result( numRows, numCols, CV_8UC1, ( void* ) data );
        cv::imwrite( "result.png", _result );
    }

}
