

#include "../helpers/LCommon.h"
#include "../helpers/img/LImage.h"

#include <cmath>
#include <iostream>
#include <string>

using namespace std;

#define DIM 500
#define SCALE 1.5
#define CONV_ITERS 100
#define DIV_CHECK_VALUE 1000

#define THREADS_BLOCK_DIM_X 32
#define THREADS_BLOCK_DIM_Y 32

struct LComplex
{
    float r;
    float i;

    __device__ LComplex( float real, float img )
    {
    	this->r = real;
    	this->i = img;
    }

    __device__ float magnitude2() 
    {
        return r * r + i * i;
    }

    __device__ LComplex operator*( const LComplex& other ) 
    {
        return LComplex( r * other.r - i * other.i, i * other.r + r * other.i );
    }

    __device__ LComplex operator+( const LComplex& other ) 
    {
        return LComplex( r + other.r, i + other.i );
    }
};

__device__ int juliaSetCorrespondance( int x, int y )
{
	float jx = SCALE * ( (float) ( DIM / 2 - x ) ) / ( DIM / 2 );
	float jy = SCALE * ( (float) ( DIM / 2 - y ) ) / ( DIM / 2 );

	LComplex _c( -0.8, 0.156 );
	LComplex _z( jx, jy );

	for ( int i = 0; i < CONV_ITERS; i++ )
	{
		_z = _z * _z + _c;
		if ( _z.magnitude2() > DIV_CHECK_VALUE )
		{
			return 0;
		}
	}

	return 1;
}

__global__ void juliaSetKernel( u8* d_img )
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;

	if ( tx >= DIM || ty >= DIM )
	{
		//return;
	}

	int _offset = tx + ty * DIM;

	int _juliaCorrespondance = juliaSetCorrespondance( tx, ty );
	d_img[_offset * 4 + 0] = 0;
	d_img[_offset * 4 + 1] = 255 * _juliaCorrespondance;
	d_img[_offset * 4 + 2] = 0;
	d_img[_offset * 4 + 3] = 255;
}

int main()
{
	img::LImageRGB _img( DIM, DIM );

	u8* d_img;

        cout << "img props: " << "w: " << _img.w << " - h: " << _img.h << endl;

	cudaMalloc( ( void** ) &d_img, sizeof( u8 ) * (_img.w * _img.h * 4 ) );

	dim3 _grid( ceil( ( (float) DIM ) / THREADS_BLOCK_DIM_X ),
				ceil( ( (float) DIM ) / THREADS_BLOCK_DIM_Y ) );

	dim3 _block( THREADS_BLOCK_DIM_X, THREADS_BLOCK_DIM_Y );

        cout << "_grid: " << _grid.x << " - " << _grid.y << endl;
        cout << "_block: " << _block.x << " - " << _block.y << endl;

	juliaSetKernel<<< _grid, _block >>>( d_img );

	cudaMemcpy( _img.buffer, d_img, sizeof( u8 ) * ( _img.w * _img.h * 4 ) , cudaMemcpyDeviceToHost );

	// save the image
	_img.saveImage( string( "julia_img.jpg" ) );

	cudaFree( d_img );

}
