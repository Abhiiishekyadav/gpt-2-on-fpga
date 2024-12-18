//#include "add.h"
//
// #define N (10 * 768)
//
//void tokenizer(
//    float wpe[N],      // Positional Encoding input
//    float wte[N],      // Token Embedding input
//    float result[N]    // Output result
//)
//{
//    #pragma HLS INTERFACE m_axi port=wpe depth=7680 offset=slave
//    #pragma HLS INTERFACE m_axi port=wte depth=7680 offset=slave
//    #pragma HLS INTERFACE m_axi port=result depth=7680 offset=slave
//    #pragma HLS INTERFACE s_axilite port=return
//
////    #pragma HLS DATAFLOW
//    #pragma HLS ARRAY_PARTITION variable=wpe cyclic factor=30
//    #pragma HLS ARRAY_PARTITION variable=wte cyclic factor=30
//    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=30
//
//    for (int i = 0; i < N; i++) {
//         #pragma HLS UNROLL
//        result[i] = wpe[i] + wte[i];
//    }
//}
#include "add.h"
#define N (10 * 768)

void tokenizer(
    float wpe[N],      // Positional Encoding input
    float wte[N],      // Token Embedding input
    float result[N]    // Output result
)
{
    #pragma HLS INTERFACE m_axi port=wpe depth=7680 offset=slave
    #pragma HLS INTERFACE m_axi port=wte depth=7680 offset=slave
    #pragma HLS INTERFACE m_axi port=result depth=7680 offset=slave
    #pragma HLS INTERFACE s_axilite port=return

    // Declare local BRAM storage
    float local_wpe[N];
    float local_wte[N];
    float local_result[N];


    // Parallel load from global memory to local BRAM
     load_wpe: for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=100
        local_wpe[i] = wpe[i];
    }

    load_wte: for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=100
        local_wte[i] = wte[i];
    }
//#pragma HLS PIPELINE II=1
    // Computation using local BRAM
     compute: for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=100
        local_result[i] = local_wpe[i] + local_wte[i];
    }

    // Store back to global memory
     store_result: for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=100
        result[i] = local_result[i];
    }
}
