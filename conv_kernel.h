#include <immintrin.h>

void print_reg(__m256d reg) {
  printf("%lf %lf %lf %lf\n", reg[0], reg[1], reg[2], reg[3]);
  printf("\n");
}

void conv_kernel
(
  double*     restrict input,
  double*     restrict filter,
  double*     restrict output,
  int    length,
  int    width
){

  __m256d r1, r2, r3, r4, r5, r6, r7, r8;
  __m256d r9, r10, r11, r12, r13, r14, r15, r16;

  r3 = _mm256_setzero_pd();  
  r4 = _mm256_setzero_pd();  
  r6 = _mm256_setzero_pd();  
  r7 = _mm256_setzero_pd();  
  r8 = _mm256_setzero_pd();  
  r9 = _mm256_setzero_pd();  
  r10 = _mm256_setzero_pd();  
  r12 = _mm256_setzero_pd();  
  r13 = _mm256_setzero_pd();  
  r14 = _mm256_setzero_pd();  
  r15 = _mm256_setzero_pd();  
  r16 = _mm256_setzero_pd();  

  // first 
  r1 = _mm256_broadcast_sd(&filter[0]);
  r2 = _mm256_broadcast_sd(&filter[2]);

  r5 = _mm256_loadu_pd(&input[width]);
  r11 = _mm256_loadu_pd(&input[4*width]);

  r16 = _mm256_fmadd_pd(r5, r2, r16);
  r15 = _mm256_fmadd_pd(r5, r1, r15);

  r12 = _mm256_fmadd_pd(r11, r1, r12);  r5 = _mm256_loadu_pd(&input[2*width]);
  r13 = _mm256_fmadd_pd(r11, r2, r13);

  r14 = _mm256_fmadd_pd(r5, r1, r14);  r11 = _mm256_loadu_pd(&input[3*width]);
  r10 = _mm256_fmadd_pd(r5, r2, r10);

  r3 = _mm256_fmadd_pd(r11, r1, r3);  r5 = _mm256_loadu_pd(&input[5*width]);
  r4 = _mm256_fmadd_pd(r11, r2, r4);

  r8 = _mm256_fmadd_pd(r5, r1, r8);  r11 = _mm256_loadu_pd(&input[6*width]);
  r6 = _mm256_fmadd_pd(r5, r2, r6);

  r5 = _mm256_loadu_pd(&input[0]);

  r7 = _mm256_fmadd_pd(r11, r2, r7);
  r9 = _mm256_fmadd_pd(r5, r1, r9);
  
  // second 
  r1 = _mm256_broadcast_sd(&filter[1]);   r5 = _mm256_loadu_pd(&input[width+1]);
  r2 = _mm256_broadcast_sd(&filter[3]);   

  r16 = _mm256_fmadd_pd(r5, r2, r16);  r11 = _mm256_loadu_pd(&input[4*width+1]);
  r15 = _mm256_fmadd_pd(r5, r1, r15);

  r12 = _mm256_fmadd_pd(r11, r1, r12); r5 = _mm256_loadu_pd(&input[2*width+1]);
  r13 = _mm256_fmadd_pd(r11, r2, r13);

  r14 = _mm256_fmadd_pd(r5, r1, r14); r11 = _mm256_loadu_pd(&input[3*width+1]);
  r10 = _mm256_fmadd_pd(r5, r2, r10);  

  r3 = _mm256_fmadd_pd(r11, r1, r3); r5 = _mm256_loadu_pd(&input[5*width+1]);
  r4 = _mm256_fmadd_pd(r11, r2, r4);

  r8 = _mm256_fmadd_pd(r5, r1, r8); r11 = _mm256_loadu_pd(&input[6*width+1]);
  r6 = _mm256_fmadd_pd(r5, r2, r6); 

  r5 = _mm256_loadu_pd(&input[1]);

  r7 = _mm256_fmadd_pd(r11, r2, r7);
  r9 = _mm256_fmadd_pd(r5, r1, r9);

  // add
  r16 = _mm256_add_pd(r16, r9);
  r15 = _mm256_add_pd(r15, r10);
  r14 = _mm256_add_pd(r14, r4);
  r13 = _mm256_add_pd(r13, r3);
  r12 = _mm256_add_pd(r12, r6);
  r8 = _mm256_add_pd(r8, r7);

  _mm256_store_pd(&output[0], r16);
  _mm256_store_pd(&output[4], r15);
  _mm256_store_pd(&output[8], r14);
  _mm256_store_pd(&output[12], r13);
  _mm256_store_pd(&output[16], r12);
  _mm256_store_pd(&output[20], r8);

}
