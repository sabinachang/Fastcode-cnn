#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "immintrin.h"
#include "conv_kernel.h"
#include "conv_pack.h"
#include "maxpool_kernel.h"
#include "maxpool_pack.h"
#include "relu_kernel.h"


#ifndef __cplusplus
#ifndef _BOOL
typedef unsigned char bool;
static const bool False = 0;
static const bool True = 1;
#endif
#endif 

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


void print_row_major(double* array, int l, int w) {
  for (int i=0; i != l; ++i) {
    for (int j=0; j != w; ++j) {
      printf("%lf ", array[i*w+j]);
    }
    printf("\n");
  }
}

void print_pack(double* pack_input, int w, int kw, int pl, bool dec_one) {
  int itr = 0;
  int step = dec_one ? kw - 1 : kw;
  for (int j= 0; j < w; j += step) {
    printf("j=%d\n", j);
    for (int i = 0; i != pl; ++i) {
      for (int k=0; k != kw; ++k) {
        printf("%lf ", pack_input[itr]);
        itr ++;
      }
      printf("\n");
    }
    printf("\n");
  }
}


int main(int argc, char **argv){
  int RUNS = atoi(argv[1]);
  int threads = atoi(argv[2]);

  /* original */
  double *input;
  double *filter;
  unsigned int w = 28; // input size
  unsigned int os = 13; // output size
  unsigned int f = 2; 
  unsigned int nf = 1; // number of filter
  double *output;
  unsigned long long t0, t1, sum0, sum1, sum2;

  posix_memalign((void**) &input, 64, w * w * sizeof(double));
  posix_memalign((void**) &filter, 64, f * f * sizeof(double));
  posix_memalign((void**) &output, 64, os * os * sizeof(double));

  /* conv layer */
  double *pack_input_conv;
  double *output_conv, *pack_output_conv;
  unsigned int o = 27;  
  unsigned int kl = 7;
  unsigned int kw = 5;
  unsigned int pl = 31;
  unsigned int pw = 35;
  unsigned int ol = 30;
  unsigned int ow = 28;

  posix_memalign((void**) &pack_input_conv, 64, pl * pw * sizeof(double));
  posix_memalign((void**) &output_conv, 64, o * ow* sizeof(double));
  posix_memalign((void**) &pack_output_conv, 64, ol * ow * sizeof(double));

  /* maxpool layer */
  double *pack_input_maxpool;
  double *input_tmp;
  double *output_maxpool;
  double *original = output_conv;

  posix_memalign((void**) &input_tmp, 64, 26 * 26 * sizeof(double));
  posix_memalign((void**) &pack_input_maxpool, 64, 26 * 26 * sizeof(double));
  posix_memalign((void**) &output_maxpool, 64, os * os * sizeof(double));
  

  /* initialize */
  int itr = 1;
  for (int i=0; i != w*w; ++i) {
    input[i] = itr; 
    itr += 1;
  }
  printf("input matrix:\n");
  print_row_major(input, w, w); 
  printf("\n"); 

  filter[0] = 0;
  filter[1] = 1;
  filter[2] = 2;
  filter[3] = 3;

  for (int i = 0; i != (13 * 13); ++i){
    output[i] = 0.0;
  }


  sum0 = 0; 
  sum1 = 0;
  sum2 = 0;
  for (int runs = 0; runs != RUNS; ++runs){

    for (int num_fiter=0; num_fiter < nf; ++num_fiter) {

      /* ---------------------------------------- run conv pack & kernel ---------------------------------------- */

      conv_pack(input, pack_input_conv, kl, kw, pl, pw, w);
      // print_pack(pack_input_conv, w, kw, pl, True);

      #pragma omp parallel num_threads(7)
      #pragma omp single
      for (int c=0; c<7; c++) {
        #pragma omp task
        {
          for (int r=0; r<5; r++) {
            t0 = rdtsc();
            conv_kernel(&pack_input_conv[c*pl*kw + r*kw*(kl-1)], filter, &pack_output_conv[c*ol*4 + r*4*(kl-1)], kl, kw);
            t1 = rdtsc();
            #pragma omp atomic
            sum0 += (t1 - t0);
          }
        }
      } 
      // print_pack(pack_output_conv, ow, 4, ol, False);
      conv_unpack(pack_output_conv, output_conv, ol, ow, o, kl);
      // print_row_major(output_conv, o, o);

      /* ---------------------------------------- run maxpool pack & kernel ---------------------------------------- */  

      for (int i = 0; i != (26 * 26); ++i){
          input_tmp[i] = 0.0;
      }

      // drop last column and last row of the original input
      // so that 2x2 stride 2 max pooling can fit completely
      // 27*27 -> 26*26
      // Note: if baseline project preprocess 27*27 image differently
      // the result could be different. Check how baseline
      // handles odd image dimension
      int input_idx = 0;
      for (int i = 0; i != (26*27); i++) {
          if (((i+1)%27) == 0) {
              continue;
          }
          input_tmp[input_idx] = original[i];
          input_idx += 1;
      }

      for (int i = 0; i != (26 * 26); ++i){
          pack_input_maxpool[i] = 0.0;
      }

      for (int i = 0; i != (13 * 13); ++i){
          output_maxpool[i] = 0.0;
      }
      
      // reorganize memory layout so that kernel can 
      // produce max pooling result stored in row major order
      maxpool_pack(input_tmp, pack_input_maxpool);

      // feed 14 panels into kernel
      #pragma omp parallel for num_threads(threads) reduction(+:sum1)
      for(int i = 0; i < 14; i++) {
        t0 = rdtsc();
        maxpool_kernel(pack_input_maxpool+(i*48),(12*i)+output_maxpool);
        t1 = rdtsc();
        sum1 += (t1 - t0);
      }

      double max = DBL_MIN;
      //complete max pool on last 2*2 block
      for (int i = 14*18; i < 676; i++) {
          if (pack_input_maxpool[i] > max) {
              max = pack_input_maxpool[i];
          }
      }
      output_maxpool[(134*13)-1] = max;
      //print_row_major(output_maxpool, 13, 13); //TODO: is the dimension 13?

      /* ---------------------------------------- run relu kernel ---------------------------------------- */  
      
      // point input to output from previous layer
      double *input_relu = output_maxpool;

      // perform relu on first 9 elements 
      relu_preprocess(input_relu, output);
      
      // run kernel 5 times to process rest of the 160 elements
      #pragma omp parallel for num_threads(threads) reduction(+:sum2)
      for (int i = 0; i < 5; i++){
        t0 = rdtsc();
        relu_kernel(input_relu+8+(i*32), output+8+(i*32));
        t1 = rdtsc();
        sum2 += (t1 - t0);
      }
      printf("output matrix:\n");
      print_row_major(output, os, os); 
      printf("\n"); 
    }

  }

  int conv = o*o*nf*f*f*2;
	int pool = os*os*nf*f*f;
	int relu = os*os*nf;
  printf("\n");
  printf("conv layer IPC: %lf\n", conv/((sum0)/(1.0*RUNS)));
  printf("pool layer IPC: %lf\n", (pool*1.0)/((sum1)/(1.0*RUNS)));
  printf("relu layer IPC: %lf\n", (relu*1.0)/((sum2)/(1.0*RUNS)));	
  printf("\n");
  printf("conv cycles: %lf\n", (double)(sum0/(1.0*RUNS)));
  printf("maxpool cycles: %lf\n", (double)(sum1/(1.0*RUNS)));
  printf("relu cycles: %lf\n", (double)(sum2/(1.0*RUNS)));

  free(input);
  free(filter);
  free(output);
  free(pack_input_conv);
  free(output_conv);
  free(pack_output_conv);
  free(input_tmp);
  free(pack_input_maxpool);
  free(output_maxpool);

  return 0;
}
