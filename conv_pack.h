// pack a row major order input into multiple overlapping columns of panel
void conv_pack(double* input, double* pack_input, int kl, int kw, int pl, int pw, int w)
{
  int itr = 0;
  for (int j=0; j < w; j+= kw - 1) {
    for (int i = 0; i != w; ++i) { 
      int start = i * w + j;
      int k_end = j+kw-1 < w ? kw : kw - 1;
      for (int k = 0; k != k_end; ++k) {
        pack_input[itr] = input[start + k];
        itr ++;
      }
      if (k_end == kw - 1) {
        itr ++;
      }
    }
    for (int i = 0; i != pl - w; ++i) {
      itr += kw;
    }
  }
}

void conv_unpack(double* pack_output, double* output, int ol, int ow, int o, int kl)
{
  int itr = 0;
  for (int r=0; r < o; ++r) {
    for (int p=0; p < 7; ++p) {
      for (int k = 0; k <4; ++k) {
        if (p != 6 || k != 3) {
          output[itr] = pack_output[p*ol*4+r*4+k];
          itr ++;
        }
      }
    }
  }
}
