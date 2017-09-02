
#pragma once

#include "Common.h"


#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#define KERNEL_WIDTH 9
#define KERNEL_SIGMA 2.0

using namespace std;

namespace IH
{

    struct CWorkingImg
    {
        // Size of imgs and buffs *******
        size_t rows;
        size_t cols;
        size_t filterDim;
        // ******************************

        // Host memory pointers *********
        uchar4* h_inputImageRGBA;
        uchar4* h_outputImageRGBA;
        float* h_filter;
        // ******************************

        // Device memory pointers *******
        uchar4* d_inputImageRGBA;
        uchar4* d_outputImageRGBA;
        float* d_filter;
        unsigned char* d_red;
        unsigned char* d_green;
        unsigned char* d_blue;
        unsigned char* d_redBlurred;
        unsigned char* d_greenBlurred;
        unsigned char* d_blueBlurred;
        // ******************************

        // ******************************
        // CV imgMat objects ************
        cv::Mat* cv_imgIn;
        cv::Mat* cv_imgOut;
        // ******************************
    };



    CWorkingImg preProcess( const std::string &filename ) 
    {
        // cudaFree(0);

        CWorkingImg _ws;

        uchar4* h_inputImageRGBA;
        uchar4* h_outputImageRGBA;
        float* h_filter;


        uchar4* d_inputImageRGBA;
        uchar4* d_outputImageRGBA;
        float* d_filter;

        unsigned char* d_red;
        unsigned char* d_green;
        unsigned char* d_blue;

        unsigned char* d_redBlurred;
        unsigned char* d_greenBlurred;
        unsigned char* d_blueBlurred;

        cv::Mat* _imgIn  = new cv::Mat();
        cv::Mat* _imgOut = new cv::Mat();

        cv::Mat _img = cv::imread( filename.c_str(), CV_LOAD_IMAGE_COLOR );
        if ( _img.empty() ) 
        {
            cout << "Couldn't open file" << endl;
            return _ws;
        }

        cv::cvtColor( _img, *_imgIn, CV_BGR2RGBA );
        _imgOut->create( _img.rows, _img.cols, CV_8UC4 );

        h_inputImageRGBA  = ( uchar4 * ) _imgIn->ptr<unsigned char>( 0 );
        h_outputImageRGBA = ( uchar4 * ) _imgOut->ptr<unsigned char>( 0 );

        const size_t _numPixels = _img.rows * _img.cols;
      
        cudaMalloc( &d_inputImageRGBA, sizeof( uchar4 ) * _numPixels );
        cudaMalloc( &d_outputImageRGBA, sizeof( uchar4 ) * _numPixels );
        
        cudaMemcpy( d_inputImageRGBA, h_inputImageRGBA, 
                    sizeof( uchar4 ) * _numPixels, 
                    cudaMemcpyHostToDevice );

        cudaMemset( d_outputImageRGBA, 0, 
                    sizeof( uchar4 ) * _numPixels );

        //create and fill the filter we will convolve with
        h_filter = new float[KERNEL_WIDTH * KERNEL_WIDTH];

        float filterSum = 0.0f; //for normalization

        for ( int r = -KERNEL_WIDTH / 2; r <= KERNEL_WIDTH / 2; ++r ) 
        {
            for ( int c = -KERNEL_WIDTH / 2; c <= KERNEL_WIDTH / 2; ++c ) 
            {
                float filterValue = expf( -( float )( c * c + r * r ) / ( 2.0f * KERNEL_SIGMA * KERNEL_SIGMA ) );
                h_filter[ ( r + KERNEL_WIDTH / 2 ) * KERNEL_WIDTH + c + KERNEL_WIDTH / 2] = filterValue;
                filterSum += filterValue;
            }
        }

        float normalizationFactor = 1.f / filterSum;

        for ( int r = -KERNEL_WIDTH / 2; r <= KERNEL_WIDTH / 2; ++r ) 
        {
            for (int c = -KERNEL_WIDTH / 2; c <= KERNEL_WIDTH / 2; ++c ) 
            {
                h_filter[ ( r + KERNEL_WIDTH / 2 ) * KERNEL_WIDTH + c + KERNEL_WIDTH / 2] *= normalizationFactor;
            }
        }

        // normal
        cudaMalloc( &d_red, sizeof( unsigned char ) * _numPixels );
        cudaMalloc( &d_green, sizeof( unsigned char ) * _numPixels );
        cudaMalloc( &d_blue, sizeof( unsigned char ) * _numPixels );

        // filter
        cudaMalloc( &d_filter, sizeof( float ) * KERNEL_WIDTH * KERNEL_WIDTH );
        cudaMemcpy( d_filter, h_filter, sizeof( float ) * KERNEL_WIDTH * KERNEL_WIDTH, cudaMemcpyHostToDevice );

        //blurred
        cudaMalloc( &d_redBlurred,    sizeof( unsigned char ) * _numPixels );
        cudaMalloc( &d_greenBlurred,  sizeof( unsigned char ) * _numPixels );
        cudaMalloc( &d_blueBlurred,   sizeof( unsigned char ) * _numPixels );
        
        cudaMemset( d_redBlurred,   0, sizeof( unsigned char ) * _numPixels );
        cudaMemset( d_greenBlurred, 0, sizeof( unsigned char ) * _numPixels );
        cudaMemset( d_blueBlurred,  0, sizeof( unsigned char ) * _numPixels );

        _ws.rows = _img.rows;
        _ws.cols = _img.cols;
        _ws.filterDim = KERNEL_WIDTH;
        
        _ws.h_inputImageRGBA    = h_inputImageRGBA;
        _ws.h_outputImageRGBA   = h_outputImageRGBA;
        _ws.h_filter            = h_filter;
        
        _ws.d_inputImageRGBA    = d_inputImageRGBA;
        _ws.d_outputImageRGBA   = d_outputImageRGBA;
        _ws.d_filter            = d_filter;

        _ws.d_red    = d_red;
        _ws.d_green  = d_green;
        _ws.d_blue   = d_blue;

        _ws.d_redBlurred    = d_redBlurred;
        _ws.d_greenBlurred  = d_greenBlurred;
        _ws.d_blueBlurred   = d_blueBlurred;
        
        _ws.cv_imgIn  = _imgIn;
        _ws.cv_imgOut = _imgOut;

        return _ws;
    }

    void postProcess( uchar4* h_outputImageRGBA,
                      size_t nRows, size_t nCols ) 
    {
        cv::Mat _imgOut( nRows, nCols, CV_8UC4, ( void* ) h_outputImageRGBA );

        cv::Mat _imageOutputBGR;
        cv::cvtColor( _imgOut, _imageOutputBGR, CV_RGBA2BGR);
        
        cv::imwrite( "./result.png" , _imageOutputBGR );
    }

    void cleanUp( CWorkingImg& ws )
    {
        cudaFree( ws.d_inputImageRGBA );
        cudaFree( ws.d_outputImageRGBA );
        cudaFree( ws.d_filter );

        cudaFree( ws.d_red );
        cudaFree( ws.d_green );
        cudaFree( ws.d_blue );
        cudaFree( ws.d_redBlurred );
        cudaFree( ws.d_greenBlurred );
        cudaFree( ws.d_blueBlurred );

        delete[] ws.h_filter;
        delete ws.cv_imgIn;
        delete ws.cv_imgOut;
    }

}