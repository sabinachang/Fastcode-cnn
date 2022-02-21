#include <immintrin.h>

void maxpool_kernel(
    double* restrict block,
    double* restrict out){
    __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    double* ptr = block;

    ymm0 = _mm256_load_pd(ptr);  ymm1 = _mm256_load_pd(ptr + 4);
    ymm2 = _mm256_load_pd(ptr + 8);  ymm3 = _mm256_load_pd(ptr + 12);
    ymm4 = _mm256_load_pd(ptr + 16);  ymm5 =_mm256_load_pd(ptr + 20);
    ymm6 = _mm256_load_pd(ptr + 24);  ymm7 = _mm256_load_pd(ptr + 28);
    ymm8 = _mm256_load_pd(ptr + 32);  ymm9 = _mm256_load_pd(ptr + 36);
    ymm10 = _mm256_load_pd(ptr + 40);  ymm11 = _mm256_load_pd(ptr + 44);

    ymm12 = _mm256_max_pd(ymm0, ymm1); 
    ymm13 = _mm256_max_pd(ymm2, ymm3); 
    ymm14 = _mm256_max_pd(ymm4, ymm5); 
    ymm15 = _mm256_max_pd(ymm6, ymm7); 
    ymm0 = _mm256_max_pd(ymm8, ymm9); 
    ymm1 = _mm256_max_pd(ymm10, ymm11); 

    ymm2 = _mm256_shuffle_pd(ymm12, ymm13, (0|(0<<1)|(0<<2)|(0<<3)));
    ymm3 = _mm256_shuffle_pd(ymm12, ymm13, (1|(1<<1)|(1<<2)|(1<<3)));
    ymm4 = _mm256_shuffle_pd(ymm14, ymm15, (0|(0<<1)|(0<<2)|(0<<3)));
    ymm5 = _mm256_shuffle_pd(ymm14, ymm15, (1|(1<<1)|(1<<2)|(1<<3)));
    ymm6 = _mm256_shuffle_pd(ymm0, ymm1, (0|(0<<1)|(0<<2)|(0<<3)));
    ymm7 = _mm256_shuffle_pd(ymm0, ymm1, (1|(1<<1)|(1<<2)|(1<<3)));

    ymm8 =  _mm256_max_pd(ymm2, ymm3); 
    ymm9 =  _mm256_max_pd(ymm4, ymm5); 
    ymm10 =  _mm256_max_pd(ymm6, ymm7); 

    // each 2*2 blocks is in column major order
    _mm256_store_pd(out, ymm8);
    _mm256_store_pd(out+4, ymm9);
    _mm256_store_pd(out+8, ymm10);
}