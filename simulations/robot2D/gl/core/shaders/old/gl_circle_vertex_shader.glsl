#version 330 core

layout ( location = 0 ) in vec3 position;

uniform mat4 u_transform;

void main()
{
	gl_Position = u_transform * vec4( position.x, position.y, position.z, 1.0f );
}