//complete code for this correct code
//#include "mha.h"
//void mha(
//	float inputs[5 * 768],  	// Input shape: 5x768
//	float weights[768 * 2304],  // Weight shape: 768x2304
//	float bias[2304],       	// Bias shape: 2304
//	float final[5 * 2304]   	// Output shape: 5x2304
//)
//{
//#pragma HLS INTERFACE m_axi port=inputs depth=3840 offset=slave
//#pragma HLS INTERFACE m_axi port=weights depth=1769472 offset=slave
//#pragma HLS INTERFACE m_axi port=bias depth=2304 offset=slave
//#pragma HLS INTERFACE m_axi port=final depth=11520 offset=slave
//#pragma HLS INTERFACE s_axilite port=return
//
//	float sum;
//	for (int batch = 0; batch < 5; batch++) {
//    	for (int out = 0; out < 2304; out++) {
//        	sum = 0.0;
//        	for (int in = 0; in < 768; in++) {
//            	// Corrected indexing for proper matrix multiplication
//            	sum += inputs[batch * 768 + in] * weights[in * 2304 + out];
//        	}
//        	final[batch * 2304 + out] = (sum + bias[out]);
//    	}
//	}
//}

//////////////////////////////////////////////////////////////////////////////////////////
#include  "mhanew.h"
#include <hls_math.h>
#include <stdio.h>
void simple_attention(
    float q[5][64],  // Query matrix
    float k[5][64],  // Key matrix
    float v[5][64],  // Value matrix
    float output[5][64]    // Output matrix
) {
    const float scale = 1.0f / hls::sqrt(64.0f);  // 1/sqrt(d_k)
    const float MASK_VALUE = -1e10f;  // Large negative value for masking
    float attention_scores[5][5];  // Store QK^T scores

    // First pass: compute max for each row for numerical stability
    float max_scores[5];
    MAX_I: for (int i = 0; i < 5; i++) {
        max_scores[i] = -INFINITY;
        MAX_J: for (int j = 0; j < 5; j++) {
            float qk_dot = 0.0f;
            MAX_DOT: for (int d = 0; d < 64; d++) {
                qk_dot += q[i][d] * k[j][d];
            }
            float scaled_score = qk_dot * scale;
            if (j > i) {  // Apply causal mask
                scaled_score += MASK_VALUE;
            }
            max_scores[i] = hls::fmax(max_scores[i], scaled_score);
        }
    }

    // Second pass: compute stable softmax
    SCORE_I: for (int i = 0; i < 5; i++) {
        float sum_exp = 0.0f;

        // Calculate exp(x - max) for numerical stability
        SCORE_J: for (int j = 0; j < 5; j++) {
            float qk_dot = 0.0f;
            DOT: for (int d = 0; d < 64; d++) {
                qk_dot += q[i][d] * k[j][d];
            }
            float scaled_score = qk_dot * scale;
            if (j > i) {  // Apply causal mask
                scaled_score += MASK_VALUE;
            }
            // Subtract max for stability
            attention_scores[i][j] = expf(scaled_score - max_scores[i]);
            sum_exp += attention_scores[i][j];
        }

        // Compute output with normalized attention scores
        OUT_D: for (int d = 0; d < 64; d++) {
            float weighted_sum = 0.0f;
            WEIGHTED: for (int j = 0; j < 5; j++) {
                weighted_sum += (attention_scores[i][j] / sum_exp) * v[j][d];
            }
            output[i][d] = weighted_sum;
        }
    }
}
//void simple_attention(
//    float q[5][64],
//    float k[5][64],
//    float v[5][64],
//    float output[5][64]
//) {
//    const float scale = 1.0f / hls::sqrt(64.0f);
//    const float MASK_VALUE = -1e10f;
//    float attention_scores[5][5];
//
//    // Debug input values
//    printf("Scale factor: %f\n", scale);
//
//    // Check for NaN or inf in input
//    for(int i = 0; i < 5; i++) {
//        for(int d = 0; d < 64; d++) {
//            if(isnan(q[i][d]) || isinf(q[i][d])) {
//                printf("NaN/Inf found in q[%d][%d]: %f\n", i, d, q[i][d]);
//            }
//            if(isnan(k[i][d]) || isinf(k[i][d])) {
//                printf("NaN/Inf found in k[%d][%d]: %f\n", i, d, k[i][d]);
//            }
//            if(isnan(v[i][d]) || isinf(v[i][d])) {
//                printf("NaN/Inf found in v[%d][%d]: %f\n", i, d, v[i][d]);
//            }
//        }
//    }
//
//    // Compute and check QK^T scores
//    for(int i = 0; i < 5; i++) {
//        for(int j = 0; j < 5; j++) {
//            float qk_dot = 0.0f;
//            for(int d = 0; d < 64; d++) {
//                qk_dot += q[i][d] * k[j][d];
//            }
//            float scaled_score = qk_dot * scale;
//            if(j > i) {
//                scaled_score += MASK_VALUE;
//            }
//
//            // Check for extremely large values before exp
//            if(scaled_score > 88.0f) {
//                printf("Warning: Very large score before exp at [%d][%d]: %f\n", i, j, scaled_score);
//            }
//
//            attention_scores[i][j] = expf(scaled_score);
//
//            // Check for overflow after exp
//            if(isinf(attention_scores[i][j])) {
//                printf("Overflow in attention_scores[%d][%d] after exp\n", i, j);
//            }
//        }
//    }
//
//    // Debug softmax normalization
//    for(int i = 0; i < 5; i++) {
//        float normalizer = 0.0f;
//        for(int j = 0; j < 5; j++) {
//            normalizer += attention_scores[i][j];
//        }
//
//        // Check for zero normalization constant
//        if(normalizer == 0.0f) {
//            printf("Warning: Zero normalizer at row %d\n", i);
//        }
//
//        for(int d = 0; d < 64; d++) {
//            float weighted_sum = 0.0f;
//            for(int j = 0; j < 5; j++) {
//                weighted_sum += (attention_scores[i][j] / normalizer) * v[j][d];
//            }
//            output[i][d] = weighted_sum;
//
//            // Check final output
//            if(isnan(output[i][d]) || isinf(output[i][d])) {
//                printf("NaN/Inf in output[%d][%d]: %f\n", i, d, output[i][d]);
//            }
//        }
//    }
//}
void mha(
    float inputs[5 * 768],      // Input shape: 5x768
    float weights[768 * 2304],  // Weight shape: 768x2304
    float bias[2304],           // Bias shape: 2304
	float dense_weights[768*768],
	float dense_bias[768],
    float final[5 * 2304]       // Output shape: 5x2304
) {
#pragma HLS INTERFACE m_axi port=inputs depth=3840 offset=slave
#pragma HLS INTERFACE m_axi port=weights depth=1769472 offset=slave
#pragma HLS INTERFACE m_axi port=bias depth=2304 offset=slave
#pragma HLS INTERFACE m_axi port=dense_weights depth=589824  offset=slave
#pragma HLS INTERFACE m_axi port=dense_bias depth=2304 offset=slave
#pragma HLS INTERFACE m_axi port=final depth=11520 offset=slave
#pragma HLS INTERFACE s_axilite port=return

     float final_buffer[5][2304];
     const int N_HEAD = 12;
       const int HEAD_DIM = 64;  // 768/12 = 64
     // Declare BRAM buffers for Q, K, V
         float query_buffer[5][768];
         float key_buffer[5][768];
         float value_buffer[5][768];


         float query_heads[12][5][64];
          float key_heads[12][5][64];
          float value_heads[12][5][64];


         // float final_atn[5][768];

    for (int batch = 0; batch < 5; batch++) {
         for (int out = 0; out < 2304; out++) {
            float sum = 0.0;

            for (int in = 0; in < 768; in++) {
                sum += inputs[batch * 768 + in] * weights[in * 2304 + out];
            }


            final_buffer[batch][out] = sum + bias[out];
        }
    }
    //split into threee with equal numbers with 5*768  each
    for (int batch = 0; batch < 5; batch++) {
             for (int i = 0; i < 768; i++) {
                 // Copy to query_buffer (first 768 elements)
                query_buffer[batch][i] = final_buffer[batch][i];

                // Copy to key_buffer (second 768 elements)
                key_buffer[batch][i] = final_buffer[batch][i + 768];

                // Copy to value_buffer (third 768 elements)
                value_buffer[batch][i] = final_buffer[batch][i + 1536];
            }
        }

    // Split Q, K, V into heads
     // Each having 12 heads of size 64 (768/12 = 64)
     for (int batch = 0; batch < 5; batch++) {

         for (int head = 0; head < N_HEAD; head++) {

             for (int j = 0; j < HEAD_DIM; j++) {

                 // Calculate the source index in the original buffers
                 int src_idx = head * HEAD_DIM + j;

                 // Split query into heads
                 query_heads[head][batch][j] = query_buffer[batch][src_idx];

                 // Split key into heads
                 key_heads[head][batch][j] = key_buffer[batch][src_idx];

                 // Split value into heads
                 value_heads[head][batch][j] = value_buffer[batch][src_idx];
             }
         }
     }

     // Process each head through attention and concatenate results
         for (int head = 0; head < N_HEAD; head++) {
             float head_output[5][64];


             // Call attention for current head
             simple_attention(
                 query_heads[head],
                 key_heads[head],
                 value_heads[head],
                 head_output
             );

             // Store head output in final_atn (concatenate)
             for (int batch = 0; batch < 5; batch++) {
                 for (int j = 0; j < HEAD_DIM; j++) {
                     // Calculate position in concatenated output
                     int dest_idx = head * HEAD_DIM + j;
                     key_buffer[batch][dest_idx] = head_output[batch][j];
                 }
             }
         }
     //  float dense_output[5][768] ;// buffer reuse
         for (int batch = 0; batch < 5; batch++) {
                for (int out = 0; out < 768; out++) {
                    float sum = 0.0f;
                    // Matrix multiplication
                    for (int in = 0; in < 768; in++) {
                        sum += key_buffer[batch][in] * dense_weights[in * 768 + out];
                    }
                    // Add bias
                    query_buffer[batch][out] = sum + dense_bias[out];
                }
            }



         // Transfer attention output to final array
          for (int batch = 0; batch < 5; batch++) {
              for (int i = 0; i < 768; i++) {
                  final[batch * 2304 + i] = query_buffer[batch][i];
              }
          }
      }




