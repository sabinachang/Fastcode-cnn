#include <immintrin.h>

void relu_preprocess(    
    double* restrict input,
    double* restrict out
    ) {

    double* ptr = input;
    // Handle first 8 elements with SIMD
    __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;
    ymm0 = _mm256_setzero_pd();  
    ymm3 = _mm256_setzero_pd();

    ymm1 = _mm256_load_pd(ptr);
    ymm4 = _mm256_load_pd(ptr+4);

    ymm2 = _mm256_max_pd(ymm0,ymm1);
    ymm5 = _mm256_max_pd(ymm3,ymm4);

    _mm256_store_pd(out, ymm2);
    _mm256_store_pd(out+4, ymm5);

    // Handle last 1 element
    double c = 0.0;
    if (input[8] > c) {
        c = input[8];
    }
    out[8] = c; //TODO: originally it was output here

}

void relu_kernel(
    double* restrict input,
    double* restrict out){
    __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14;

    double* ptr = input;

    
    ymm0 = _mm256_setzero_pd();  
    ymm1 = _mm256_setzero_pd();
    ymm2 = _mm256_setzero_pd();  
       
    
    ymm3 = _mm256_load_pd(ptr + 0);
    ymm4 = _mm256_load_pd(ptr + 4); 
    ymm5 =_mm256_load_pd(ptr + 8);
    ymm6 = _mm256_load_pd(ptr + 12);  
    ymm7 = _mm256_load_pd(ptr + 16);
    ymm8 = _mm256_load_pd(ptr + 20);  
    ymm9 = _mm256_load_pd(ptr + 24);
    ymm10 = _mm256_load_pd(ptr + 28);  

    ymm11 = _mm256_max_pd(ymm0,ymm3);
    ymm12 = _mm256_max_pd(ymm1,ymm4);
    ymm13 = _mm256_max_pd(ymm2,ymm5);

    ymm14 = _mm256_max_pd(ymm0, ymm6); _mm256_store_pd(out, ymm11);
    ymm11 = _mm256_max_pd(ymm1, ymm7); _mm256_store_pd(out+4, ymm12);
    ymm12 = _mm256_max_pd(ymm2, ymm8); _mm256_store_pd(out+8, ymm13);
    ymm13 = _mm256_max_pd(ymm2, ymm9); _mm256_store_pd(out+12, ymm14);
    ymm14= _mm256_max_pd(ymm2, ymm10); _mm256_store_pd(out+16, ymm11);

    _mm256_store_pd(out+20, ymm12);
    _mm256_store_pd(out+24, ymm13);
    _mm256_store_pd(out+28, ymm14);
}