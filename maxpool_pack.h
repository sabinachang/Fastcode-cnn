#include <immintrin.h>

void maxpool_pack(double *input, double *pack) {

    __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    double *tmp;
    // tmp holds 48 elements from input whose
    // memory location is reorganized to fit the kernel size
    posix_memalign((void **) &tmp, 64, 48*sizeof(double));
    for (int i = 0; i != 48; ++i){
        tmp[i] = 0.0;
    }
    double *pptr = pack;

    int prev_idx = 0;
    int cur_idx = 0;

    // there are 14 panels that will be fed into the kernel
    for (int i = 0; i < 14; i++ ) {

        // get the correct 48 elements from input and put them in temp
        double *ptr = tmp;
        int t = 0;

        //no elements left in previous row, reset cur_idx prev_idx
        if ((cur_idx - prev_idx) > 24) {
            prev_idx = prev_idx - 26;
            cur_idx = prev_idx;
        }

        // This part handles elemnents left to be processed in the
        // previous row
        for (int k = prev_idx; k < cur_idx; k++) {
            tmp[t] = input[k-26];
            tmp[t+1] = input[k];

            t += 2;
        }

        // num of elements current row need to process
        int count = 24- (cur_idx - prev_idx);
        
        for (int l = cur_idx; l <(cur_idx+count); l++) {
            tmp[t] = input[l];
            tmp[t+1] = input[l+26]; 

            t += 2;
        }

        //move prev_idx to where cur_idx left off
        prev_idx = cur_idx + count+26;
        // move cur_idx to next row
        cur_idx = 2*(i+1)*26;

        // shuffle the 48 elements to correct position
        // so that after maxpool, elements are still in
        // row major order
        ymm15 = _mm256_load_pd(ptr); 
        ymm14 = _mm256_load_pd(ptr+4);
        ymm13 = _mm256_load_pd(ptr+8);
        ymm12 = _mm256_load_pd(ptr+12); 

        ymm11 = _mm256_permute4x64_pd(ymm15,(0|(2<<2)|(1<<4)|(3<<6)));
        ymm10 = _mm256_permute4x64_pd(ymm14,(0|(2<<2)|(1<<4)|(3<<6)));
        ymm9 = _mm256_permute4x64_pd(ymm13,(0|(2<<2)|(1<<4)|(3<<6)));
        ymm8 = _mm256_permute4x64_pd(ymm12,(0|(2<<2)|(1<<4)|(3<<6)));

        ymm7 = _mm256_permute2f128_pd(ymm11, ymm9, (0|(2<<4)));
        ymm6 = _mm256_permute2f128_pd(ymm11, ymm9, (1|(3<<4)));
        ymm5 = _mm256_permute2f128_pd(ymm10, ymm8, (0|(2<<4)));
        ymm4 = _mm256_permute2f128_pd(ymm10, ymm8, (1|(3<<4)));

        _mm256_store_pd(pptr, ymm7);
        _mm256_store_pd(pptr+4, ymm6);
        _mm256_store_pd(pptr+8, ymm5);
        _mm256_store_pd(pptr+12, ymm4);

        ymm3 = _mm256_load_pd(ptr+16);
        ymm2 = _mm256_load_pd(ptr+20); 
        ymm1 = _mm256_load_pd(ptr+24); 
        ymm0 = _mm256_load_pd(ptr+28); 

        ymm15 = _mm256_permute4x64_pd(ymm3,(0|(2<<2)|(1<<4)|(3<<6)));
        ymm14 = _mm256_permute4x64_pd(ymm2,(0|(2<<2)|(1<<4)|(3<<6)));
        ymm13 = _mm256_permute4x64_pd(ymm1,(0|(2<<2)|(1<<4)|(3<<6)));
        ymm12 = _mm256_permute4x64_pd(ymm0,(0|(2<<2)|(1<<4)|(3<<6)));

        ymm11 = _mm256_permute2f128_pd(ymm15, ymm13, (0|(2<<4)));
        ymm10 = _mm256_permute2f128_pd(ymm15, ymm13, (1|(3<<4)));
        ymm9 = _mm256_permute2f128_pd(ymm14, ymm12, (0|(2<<4)));
        ymm8 = _mm256_permute2f128_pd(ymm14, ymm12, (1|(3<<4)));

        _mm256_store_pd(pptr+16, ymm11);
        _mm256_store_pd(pptr+20, ymm10);
        _mm256_store_pd(pptr+24, ymm9);
        _mm256_store_pd(pptr+28, ymm8);

        ymm15 = _mm256_load_pd(ptr+32); 
        ymm14 = _mm256_load_pd(ptr+36);
        ymm13 = _mm256_load_pd(ptr+40);
        ymm12 = _mm256_load_pd(ptr+44); 

        ymm11 = _mm256_permute4x64_pd(ymm15,(0|(2<<2)|(1<<4)|(3<<6)));
        ymm10 = _mm256_permute4x64_pd(ymm14,(0|(2<<2)|(1<<4)|(3<<6)));
        ymm9 = _mm256_permute4x64_pd(ymm13,(0|(2<<2)|(1<<4)|(3<<6)));
        ymm8 = _mm256_permute4x64_pd(ymm12,(0|(2<<2)|(1<<4)|(3<<6)));

        ymm7 = _mm256_permute2f128_pd(ymm11, ymm9, (0|(2<<4)));
        ymm6 = _mm256_permute2f128_pd(ymm11, ymm9, (1|(3<<4)));
        ymm5 = _mm256_permute2f128_pd(ymm10, ymm8, (0|(2<<4)));
        ymm4 = _mm256_permute2f128_pd(ymm10, ymm8, (1|(3<<4)));

        _mm256_store_pd(pptr+32, ymm7);
        _mm256_store_pd(pptr+36, ymm6);
        _mm256_store_pd(pptr+40, ymm5);
        _mm256_store_pd(pptr+44, ymm4);

        // prepare to store next 48 elements
        pptr += 48;
    }

    //put the last 2*2 block into pack
    for (int i = prev_idx; i < 26*26; i++) {
        *pptr = input[i-26];
        *(pptr+1) = input[i];

        pptr += 2;
    }
}