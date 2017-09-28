

typedef struct
{
	float p1x;
	float p2x;
	float nx;
	float ny;
	float ux;
	float uy;
	float len;
} LLine;


typedef struct
{
	float x;
	float y;
	float vx;
	float vy;
	float r;
} LParticle;


__kernel void kn_collision_boundaries( __global LLine* d_lines,
									   __global LParticle* d_particles,
									   int nLines, int nParticles )
{
	int _indx = get_global_id( 0 );

	if ( _indx >= nParticles )
	{
		return;
	}

	if ( d_particles[_indx].vx == 0 && d_particles[_indx].vy == 0 )
	{
		return;
	}

	for ( int q = 0; q < nLines; q++ )
	{

		// first check if the point is in the inner side of the line
		float dx = d_particles[_indx].x - d_lines[q].p1x;
		float dy = d_particles[_indx].y - d_lines[q].p1y;
		float len = sqrt( dx * dx + dy * dy );
		float ux = dx / len;
		float uy = dy / len;

		float dot_v_normal = line.nx * ux + line.ny * uy;
		if ( dot_v_normal >= 0 )
		{
			// is in the inside region of the line, so no collision
			continue;
		}

		// Check collision point
		float _ul_x = d_lines[q].ux;
		float _ul_y = d_lines[q].uy;

		float _up_x = d_particles[_indx].vx;
		float _up_y = d_particles[_indx].vy;
		float _vp = sqrt( _up_x * _up_x + _up_y * _up_y );
		_up_x = _up_x / _vp;
		_up_y = _up_y / _vp;

		float _det = _up_x * _ul_y - _up_y * _ul_x;

		float _t = ( -_up_y * dx + _up_x * dy ) / _det;
		float _q = ( -_ul_y * dx + _ul_x * dy ) / _det;

		if ( _t <= 0 || _t >= d_lines[q].len || _q > 0 )
		{
			// If ray touches the line outside the segment or ...
			// if the ray is in the positive mov direction of the particle
			return;
		}

		// Must have hit this wall, calculate back-return distance

		float _cos = _up_x * _ul_x + _up_y * _ul_y;
		float _sin = sqrt( 1 - _cos * _cos );
		float _back_ret_dist = abs( _q ) + ( d_particles[_indx].r / _sin );

		// return the particle to the place where it just hits the wall
		d_particles[_indx].x -= _up_x * _back_ret_dist;
		d_particles[_indx].y -= _up_y * _back_ret_dist;

		// Apply elastic hit
		float _vt = d_particles[_indx].vx * d_lines[q].ux + d_particles[_indx].vy * d_lines[q].uy;
		float _vn = d_particles[_indx].vx * d_lines[q].nx + d_particles[_indx].vy * d_lines[q].ny;
		_vn *= -1;

		d_particles[_indx].vx = d_lines[q].ux * _vt + d_lines[q].nx * _vn;
		d_particles[_indx].vy = d_lines[q].uy * _vt + d_lines[q].ny * _vn;

	}

}