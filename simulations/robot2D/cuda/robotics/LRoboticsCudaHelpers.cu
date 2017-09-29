
#include "LRoboticsCudaHelpers.h"

#include <iostream>

using namespace std;

__global__ void rb_init_random_generator( u32 seed, 
										  curandState_t* pCurandStates,
										  int nParticles )
{
	int kIdx = threadIdx.x + blockIdx.x + blockDim.x;
	if ( kIdx < nParticles )
	{
		curand_init( seed,
					 kIdx,
					 0,
					 &pCurandStates[kIdx] );
	}
}

__device__ float rb_sample_normal_distribution( int kIdx, curandState_t* pCurandStates,
												float sigma2 )
{
	float _res = curand_normal( pCurandStates + kIdx );
	_res = 2 * ( _res - 0.5f ) * sigma2;
	return _res;
}


__global__ void rb_update_particle( CuParticle* pParticles, int nParticles,
									curandState_t* pCurandStates,
									float dt, float v, float w )
{
	int kIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if ( kIdx >= nParticles )
	{
		return;
	}

    float vv = v + pParticles[kIdx].d1;

    float ww = w + pParticles[kIdx].d2;

    float rt = pParticles[kIdx].d3;

    float x = pParticles[kIdx].x;
    float y = pParticles[kIdx].y;
    float t = pParticles[kIdx].t;

    if ( abs( ww ) < 0.001f )
    {
        x += vv * dt * cosf( t );
        y += vv * dt * sinf( t );
    }
    else
    {
        x += ( vv / ww ) * ( sinf( t + ww * dt ) - sinf( t ) );
        y += ( vv / ww ) * ( -cosf( t + ww * dt ) + cosf( t ) );
        t += ww * dt + rt * dt;
    }

    pParticles[kIdx].x = x;
    pParticles[kIdx].y = y;
    pParticles[kIdx].t = t;    
}

__global__ void rb_update_particle_weight_3d_separation( CuParticle* d_particles, int nParticles,
										   				 CuLine* d_lines, int nLines,
										   				 float* d_sensorsZ, float* d_sensorsAng, int nSensors )
{


}

__global__ void rb_raycast_particle( CuParticle* d_particles, int nParticles,
									 CuLine* d_lines, int nLines,
									 float* d_sensorsZ, float* d_sensorsAng, int nSensors,
									 int indxLine, int indxSensor )
{
	int pIndx = threadIdx.x + blockIdx.x * blockDim.x;

	if ( pIndx >= nParticles )
	{
		return;
	}

	float _pr_x = d_particles[pIndx].x;
	float _pr_y = d_particles[pIndx].y;

	float _p1_x = d_lines[indxLine].p1x;
	float _p1_y = d_lines[indxLine].p1y;

	float _dx = d_lines[indxLine].p2x - _p1_x;
	float _dy = d_lines[indxLine].p2y - _p1_y;
	float _dlen = sqrtf( _dx * _dx + _dy * _dy );

	float _ul_x = _dx / _dlen;
	float _ul_y = _dy / _dlen;

	float _ur_x = cosf( d_particles[pIndx].t + d_sensorsAng[indxSensor] );
	float _ur_y = sinf( d_particles[pIndx].t + d_sensorsAng[indxSensor] );

	float _det = _ur_x * _ul_y - _ur_y * _ul_x;

	float _dx_rl = _pr_x - _p1_x;
	float _dy_rl = _pr_y - _p1_y;

	float _t = ( -_ur_y * _dx_rl + _ur_x * _dy_rl ) / _det;
	float _q = ( -_ul_y * _dx_rl + _ul_x * _dy_rl ) / _det;

	if ( _t > 0 && _t < _dlen )
	{
		if ( d_particles[pIndx].rayZ[indxSensor] > _q && _q > 0 )
		{
			d_particles[pIndx].rayZ[indxSensor] = _q;
		}
	}
}

__global__ void rb_update_particle_weight( CuParticle* d_particles, int nParticles,
										   float* d_sensorsZ, int nSensors )
{
	int pIndx = threadIdx.x + blockIdx.x * blockDim.x;

	if ( pIndx >= nParticles )
	{
		return;
	}

	for ( int s = 0; s < nSensors; s++ )
	{
		float _dz = d_particles[pIndx].rayZ[s] - d_sensorsZ[s];
		float _w = ( 1.0f / sqrtf( 2 * PI * SIGMA_SENSOR * SIGMA_SENSOR ) ) *
						expf( -0.5 * _dz * _dz / ( SIGMA_SENSOR * SIGMA_SENSOR ) );

		d_particles[pIndx].wz *= _w;
	}
}

/*
 In this part we consider the particles as a 1d direction computation ...
 as far as kernels are concerned. In other parts of the complete algorithm ...
 we will consider the computation as 2D or 3D ( particles - sensors - wallLines )
*/
void rb_pf_motion_model_step( CuParticle* h_particles, int nParticles, 
							  float dt, float v, float w )
{
	// Calculate the computation separation **************************
	int nThreadsPerBlock = CU_THREADS_PER_BLOCK;
	if ( nParticles % nThreadsPerBlock != 0 )
	{
		cout << "rb_pf_motion_model_step> nParticles: " << nParticles << " should be multiple of " << nThreadsPerBlock 
			 << " for better perfomance" << endl;
	}

	int nBlocksPerGrid  = ceil( ( (float) nParticles ) / nThreadsPerBlock );
	// ***************************************************************

	// Initliaze the required memory in GPU ******************
	curandState_t* d_curandStates;
	cudaMalloc( ( void** )&d_curandStates, sizeof( curandState_t ) * nParticles );

	CuParticle* d_particles;
	cudaMalloc( ( void** )&d_particles, sizeof( CuParticle ) * nParticles );
	cudaMemcpy( d_particles, h_particles, sizeof( CuParticle ) * nParticles, cudaMemcpyHostToDevice );
	// *******************************************************

	// Launch kernels *************************************************

	// Initialize the random number generator
	rb_init_random_generator<<< nBlocksPerGrid, nThreadsPerBlock >>>( 0, d_curandStates, nParticles );

	// Update particles
	rb_update_particle<<< nBlocksPerGrid, nThreadsPerBlock >>>( d_particles, nParticles, d_curandStates,
																dt, v, w );

	// ****************************************************************

	// Retrieve results
	cudaMemcpy( h_particles, d_particles, sizeof( CuParticle ) * nParticles, cudaMemcpyDeviceToHost );
}


/*
 In this part we consider the problem as a 3d direction computation.
 Particles correspond to the x direction of computation, lines to the y direction, and ...
 sensors to the z direction of computation
*/
void rb_pf_sensor_model_step( CuParticle* h_particles, int nParticles,
                              CuLine* h_lines, int nLines,
                              float* h_sensorsZ, float* h_sensorsAng, int nSensors )
{

	// Calculate the computation separation **************************
	int nThreadsPerBlockX = CU_THREADS_PER_BLOCK;
	if ( nParticles % nThreadsPerBlockX != 0 )
	{
		cout << "rb_pf_sensor_model_step> nParticles: " << nParticles << " should be multiple of " << nThreadsPerBlockX 
			 << " for better perfomance" << endl;
	}

	int nBlocksPerGridX  = ceil( ( (float) nParticles ) / nThreadsPerBlockX );

	#ifdef CU_USE_3D_SEPARATION

	dim3 _grid( nBlocksPerGridX, nLines, nSensors );
	dim3 _block( nThreadsPerBlockX, 1, 1 );

	#else

	dim3 _grid( nBlocksPerGridX, 1, 1 );
	dim3 _block( nThreadsPerBlockX, 1, 1 );

	#endif
	// ***************************************************************

	// Initliaze the required memory in GPU ******************
	CuParticle* d_particles;
	cudaMalloc( ( void** )&d_particles, sizeof( CuParticle ) * nParticles );
	cudaMemcpy( d_particles, h_particles, sizeof( CuParticle ) * nParticles, cudaMemcpyHostToDevice );

	CuLine* d_lines;
	cudaMalloc( ( void** )&d_lines , sizeof( CuLine ) * nLines );
	cudaMemcpy( d_lines, h_lines, sizeof( CuLine ) * nLines, cudaMemcpyHostToDevice );

	float* d_sensorsZ;
	float* d_sensorsAng;
	cudaMalloc( ( void** )&d_sensorsZ, sizeof( float ) * nSensors );
	cudaMalloc( ( void** )&d_sensorsAng, sizeof( float ) * nSensors );
	cudaMemcpy( d_sensorsZ, h_sensorsZ, sizeof( float ) * nSensors, cudaMemcpyHostToDevice );
	cudaMemcpy( d_sensorsAng, h_sensorsAng, sizeof( float ) * nSensors, cudaMemcpyHostToDevice );

	// *******************************************************

	// Launch kernels *************************************************

	#ifdef CU_USE_3D_SEPARATION

	// Update particles
	rb_update_particle_weight_3d_separation<<< _grid, _block >>>( d_particles, nParticles,
													d_lines, nLines,
													d_sensorsZ, d_sensorsAng, nSensors );

	#else

	for ( int l = 0; l < nLines; l++ )
	{
		for ( int s = 0; s < nSensors; s++ )
		{
			rb_raycast_particle<<< _grid, _block >>>( d_particles, nParticles,
													  d_lines, nLines,
													  d_sensorsZ, d_sensorsAng, nSensors,
													  l, s );
		}
	}

	rb_update_particle_weight<<< _grid, _block >>>( d_particles, nParticles,
													d_sensorsZ, nSensors );
			
	#endif

	// ****************************************************************

	// Retrieve results
	cudaMemcpy( h_particles, d_particles, sizeof( CuParticle ) * nParticles, cudaMemcpyDeviceToHost );

}