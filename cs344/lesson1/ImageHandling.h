
#pragma once

#include "Common.h"


#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

namespace IH
{

  
    #ifdef USE_CUDA
    void preProcess( CEnvironment &env,
                     uchar4 **h_inputImage, u8 **h_greyImage,
                     uchar4 **d_rgbaImage, u8 **d_greyImage,
                     size_t &nRows, size_t &nCols,
                     const std::string &filename )
    #elif USE_OPENCL
    void preProcess( CEnvironment &env,
                     uchar4 **h_inputImage, u8 **h_greyImage,
                     cl::Buffer* d_rgbaBuffer, cl::Buffer* d_greyBuffer,
                     size_t &nRows, size_t &nCols,
                     const std::string &filename )
    #endif
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
        //_imgGrey->create( _img.rows, _img.cols, CV_8UC1 );
        *_imgGrey = cv::Mat::zeros( _img.rows, _img.cols, CV_8UC1 );
        nRows = _img.rows;
        nCols = _img.cols;

        *h_inputImage = ( uchar4 * ) _imgRGBA->ptr<u8>( 0 );
        *h_greyImage = _imgGrey->ptr<u8>( 0 );

        size_t _numPixels = _imgRGBA->rows * _imgRGBA->cols;
      
        cout << "allocating in device" << endl;
 
        #ifdef USE_CUDA
 
        cudaMalloc( d_rgbaImage, sizeof( uchar4 ) * _numPixels );
        cudaMalloc( d_greyImage, sizeof( u8 ) * _numPixels );
        cudaMemset( *d_greyImage, 0, sizeof( u8 ) * _numPixels );

        cudaMemcpy( *d_rgbaImage, *h_inputImage, sizeof( uchar4 ) * _numPixels, cudaMemcpyHostToDevice );

        #elif USE_OPENCL

        cout << "rgb buffer" << endl;
        cl::Buffer _d_buff_rgbaImage( env.context,
                                      CL_MEM_READ_ONLY |
                                      CL_MEM_HOST_NO_ACCESS |
                                      CL_MEM_COPY_HOST_PTR,
                                      sizeof( uchar4 ) * _numPixels,
                                      *h_inputImage );
    
        cout << "grey buffer" << endl;
        cl::Buffer _d_buff_greyImage( env.context,
                                      CL_MEM_WRITE_ONLY |
                                      CL_MEM_HOST_READ_ONLY,
                                      sizeof( u8 ) * _numPixels );
        
        *d_rgbaBuffer = _d_buff_rgbaImage;
        *d_greyBuffer = _d_buff_greyImage;

        #endif

        cout << "finished allocating in device" << endl;
    }
                     

    void postProcess( size_t numRows, size_t numCols, 
                      u8* h_greyImage, u8* d_greyImage )
    {
        #ifdef USE_CUDA

        size_t _numPixels = numRows * numCols;
        
        cudaMemcpy( h_greyImage, d_greyImage, 
                    sizeof( u8 ) * _numPixels,
                    cudaMemcpyDeviceToHost );
            
        #elif USE_OPENCL
        // Do nothing, the previous step already copied the data from the ...
        // device buffer to the host buffer
        #endif
        
        cv::Mat _result( numRows, numCols, CV_8UC1, ( void* ) h_greyImage );
        cv::imwrite( "result.png", _result );
    }


}




