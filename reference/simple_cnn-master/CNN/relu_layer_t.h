#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct relu_layer_t
{
	layer_type type = layer_type::relu;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	unsigned long long t0, t1, sum;


	relu_layer_t( tdsize in_size )
		:
		in( in_size.x, in_size.y, in_size.z ),
		out( in_size.x, in_size.y, in_size.z ),
		grads_in( in_size.x, in_size.y, in_size.z )
	{
	}


	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		// t0 = rdtsc();
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					float v = in( i, j, z );
					if ( v < 0 )
						v = 0;
					out( i, j, z ) = v;
				}


		// t1 = rdtsc();

		// sum+= (t1-t0);
	}

	void fix_weights()
	{

	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					grads_in( i, j, z ) = (in( i, j, z ) < 0) ?
						(0) :
						(1 * grad_next_layer( i, j, z ));
				}
	}

	// static __inline__ unsigned long long rdtsc(void) {
  	// unsigned hi, lo;
  	// __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  	// return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
	// }
};
#pragma pack(pop)