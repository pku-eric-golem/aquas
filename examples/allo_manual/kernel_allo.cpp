
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
using namespace std;

extern "C" {

/// This is top function.
void gemm_fp32(
  float *v0,
  float *v1,
  float *v2,
  uint8_t v3,
  float *v4
) {	// L3
  #pragma HLS interface m_axi port=v0 offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=v1 offset=slave bundle=gmem1
  #pragma HLS interface m_axi port=v2 offset=slave bundle=gmem2
  #pragma HLS interface m_axi port=v4 offset=slave bundle=gmem3
  float acc[8][8];	// L4
  #pragma HLS array_partition variable=acc complete dim=1
  #pragma HLS array_partition variable=acc complete dim=2

  for (int v6 = 0; v6 < 8; v6++) {	// L6
    for (int v7 = 0; v7 < 8; v7++) {	// L6
      acc[v6][v7] = (float)0.000000;	// L6
    }
  }
  l_mac_i: for (int i = 0; i < 8; i++) {	// L7
    float v9[8];	// L8
    l_j_init: for (int j_init = 0; j_init < 8; j_init++) {	// L9
    #pragma HLS pipeline II=1
      v9[j_init] = (float)0.000000;	// L10
    }
    l_S_k_0_k: for (int k = 0; k < 8; k++) {	// L12
      l_j: for (int j = 0; j < 8; j++) {	// L13
      #pragma HLS pipeline II=1
        float v13 = v0[((i * 8) + k)];	// L14
        float v14 = v1[((j * 8) + k)];	// L15
        float v15 = v13 * v14;	// L16
        float product;	// L17
        product = v15;	// L18
        float v17 = v9[j];	// L19
        float v18 = product;	// L20
        float v19 = v17 + v18;	// L21
        v9[j] = v19;	// L22
      }
    }
    l_j_back: for (int j_back = 0; j_back < 8; j_back++) {	// L25
    #pragma HLS pipeline II=1
      float v21 = v9[j_back];	// L26
      acc[i][j_back] = v21;	// L27
    }
  }
  l_outputs_u: for (int u = 0; u < 8; u++) {	// L30
    l_v: for (int v = 0; v < 8; v++) {	// L31
    #pragma HLS pipeline II=1
      float initial;	// L32
      initial = (float)0.000000;	// L33
      int32_t v25 = v3;	// L34
      bool v26 = v25 != 0;	// L36
      if (v26) {	// L37
        float v27 = v2[((u * 8) + v)];	// L38
        initial = v27;	// L39
      }
      float v28 = acc[u][v];	// L41
      float v29 = initial;	// L42
      float v30 = v28 + v29;	// L43
      v4[((u * 8) + v)] = v30;	// L44
    }
  }
}


} // extern "C"
