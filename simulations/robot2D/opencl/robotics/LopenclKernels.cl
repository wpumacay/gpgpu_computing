

typedef struct
{

    float x;
    float y;
    float t;

    float d1;
    float d2;
    float d3;

    float wz;
    float rayZ[20];
} ClParticle;

typedef struct
{
    float p1x;
    float p1y;
    float p2x;
    float p2y;

} ClLine;




__kernel void rb_update_particle( __global ClParticle* pParticles, 
								  int nParticles,
								  float dt, float v, float w )
{
	int _indx = get_local_id( 0 ) + get_local_size( 0 ) * get_group_id( 0 );

	if ( _indx >= nParticles )
	{
		return;
	}

    float vv = v + pParticles[_indx].d1;

    float ww = w + pParticles[_indx].d2;

    float rt = pParticles[_indx].d3;

    float x = pParticles[_indx].x;
    float y = pParticles[_indx].y;
    float t = pParticles[_indx].t;

    if ( ww < 0.001f && ww > -0.001f )
    {
        x += vv * dt * cos( t );
        y += vv * dt * sin( t );
    }
    else
    {
        x += ( vv / ww ) * ( sin( t + ww * dt ) - sin( t ) );
        y += ( vv / ww ) * ( -cos( t + ww * dt ) + cos( t ) );
        t += ww * dt + rt * dt;
    }

    pParticles[_indx].x = x;
    pParticles[_indx].y = y;
    pParticles[_indx].t = t;   
}

__kernel void rb_raycast_particle( __global ClParticle* d_particles, int nParticles,
								   __global ClLine* d_lines, int nLines,
								   __global float* d_sensorsZ, __global float* d_sensorsAng, int nSensors,
								   int indxLine, int indxSensor )
{
	int pIndx = get_local_id( 0 ) + get_local_size( 0 ) * get_group_id( 0 );

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
	float _dlen = sqrt( _dx * _dx + _dy * _dy );

	float _ul_x = _dx / _dlen;
	float _ul_y = _dy / _dlen;

	float _ur_x = cos( d_particles[pIndx].t + d_sensorsAng[indxSensor] );
	float _ur_y = sin( d_particles[pIndx].t + d_sensorsAng[indxSensor] );

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

__kernel void rb_update_particle_weight( __global ClParticle* d_particles, int nParticles,
										 __global float* d_sensorsZ, int nSensors )
{
	int pIndx = get_local_id( 0 ) + get_local_size( 0 ) * get_group_id( 0 );

	if ( pIndx >= nParticles )
	{
		return;
	}

	for ( int s = 0; s < nSensors; s++ )
	{
		float _dz = d_particles[pIndx].rayZ[s] - d_sensorsZ[s];
		float _w = ( 1.0f / sqrt( 2 *  3.1415926 * 10.0 * 10.0 ) ) *
						exp( -0.5 * _dz * _dz / ( 10.0 * 10.0 ) );

		d_particles[pIndx].wz *= _w;
	}
}
