#include "norm.h"
#include "mlp.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BATCH_SIZE 5
#define IN_FEATURES 768
#define HIDDEN_FEATURES 3072
#define OUT_FEATURES 768
#define PI 3.14159265358979323846f
#define SQRT_2_DIV_PI 0.797884560802865f
////////////////////////////////////////////////////layer norm
void layer_norm(
    float x[5 * 768],
    float gamma[768],
    float beta[768],
    float output[5 * 768],
    float eps = 1e-5
)
{


    for (int seq = 0; seq < 5; seq++) {
       // #pragma HLS PIPELINE II=1
        float mean = 0.0f, var = 0.0f;
        int base_idx = seq * 768;

        for (int i = 0; i < 768; i++) {
            mean += x[base_idx + i];
        }
        mean /= 768.0f;

        for (int i = 0; i < 768; i++) {
            float diff = x[base_idx + i] - mean;
            var += diff * diff;
        }
        var = sqrtf(var/768.0f + eps);

        for (int i = 0; i < 768; i++) {
            output[base_idx + i] = gamma[i] * ((x[base_idx + i] - mean) / var) + beta[i];
        }
    }
}

float gelu(float x) {
    float x_cubed = x * x * x;
    float inner = SQRT_2_DIV_PI * (x + 0.044715f * x_cubed);
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}
float weights1[IN_FEATURES * HIDDEN_FEATURES];   // Weights for first linear layer (768 x 3072)
float weights1_bias[HIDDEN_FEATURES];             // Bias for first linear layer (3072)
float weights2[HIDDEN_FEATURES * OUT_FEATURES];   // Weights for second linear layer (3072 x 768)
float weights2_bias[OUT_FEATURES];
// Addition function
void addition(
    float input1[BATCH_SIZE * OUT_FEATURES],
    float input2[BATCH_SIZE * OUT_FEATURES],
    float output_buffer[BATCH_SIZE * OUT_FEATURES]  // Keep output_buffer as 1D
)
{
//#pragma HLS INLINE
    ADD_BATCH: for (int b = 0; b < BATCH_SIZE; b++) {
        ADD_FEATURE: for (int j = 0; j < OUT_FEATURES; j++) {
//            #pragma HLS PIPELINE II=1
            output_buffer[b * OUT_FEATURES + j] = input1[b * OUT_FEATURES + j] + input2[b * OUT_FEATURES + j];
        }
    }
}


void ffn(
    float inputs[BATCH_SIZE * IN_FEATURES],    // Input: (5,768)
	   float ln2_b[768],
	    float ln2_g[768],
		float bias[3072],
		float  weights[2359296],
    float final[BATCH_SIZE * OUT_FEATURES]     // Output: (5,768)
)
{
#pragma HLS INTERFACE m_axi port=inputs depth=3840 offset=slave
#pragma HLS INTERFACE m_axi port=final depth=3840 offset=slave
#pragma HLS INTERFACE m_axi port=ln2_b depth=768 offset=slave
#pragma HLS INTERFACE m_axi port=ln2_g depth=768 offset=slave
#pragma HLS INTERFACE m_axi port=bias depth=3072 offset=slave
#pragma HLS INTERFACE m_axi port=weights depth=2359296 offset=slave
#pragma HLS INTERFACE s_axilite port=return

	 float normalized_inputs[BATCH_SIZE * IN_FEATURES];   //temporary buffer for normalized output



    float input_buffer[BATCH_SIZE][IN_FEATURES];
    float hidden_buffer[BATCH_SIZE][HIDDEN_FEATURES];
    float output_buffer[BATCH_SIZE * OUT_FEATURES]; // Change to 1D



    layer_norm(inputs, ln2_g, ln2_b, normalized_inputs);

//
//    for (int i = 0; i < BATCH_SIZE * IN_FEATURES; i++) {
//         final[i] = normalized_inputs[i]; // Copy the normalized values directly to final
//     }
// }

    /////////////////////////////////////////////////////////////////testing layer norm
    // Load input data into buffer
    INPUT_LOAD: for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < IN_FEATURES; j++) {
             input_buffer[i][j] = normalized_inputs[i * IN_FEATURES + j];
        }
    }

    // First linear layer: IN_FEATURES -> HIDDEN_FEATURES (768 -> 3072)
    BATCH_LOOP_1: for (int b = 0; b < BATCH_SIZE; b++) {
        OUT_LOOP_1: for (int j = 0; j < HIDDEN_FEATURES; j++) {
             float sum = bias[j];
            IN_LOOP_1: for (int k = 0; k < IN_FEATURES; k++) {
                 sum += input_buffer[b][k] * weights[k * HIDDEN_FEATURES + j];
            }
            hidden_buffer[b][j] = gelu(sum);
        }
    }

    // Copy data from hidden_buffer to final (flattened output)
     OUTPUT_STORE: for (int i = 0; i < BATCH_SIZE; i++) {
         for (int j = 0; j < OUT_FEATURES; j++) {
             // Flatten hidden_buffer (BATCH_SIZE x OUT_FEATURES) into final array (BATCH_SIZE * OUT_FEATURES)
             final[i * OUT_FEATURES + j] = hidden_buffer[i][j];
         }
     }
}
    //////////////////////////////////////////////////////////////////////////////////////////to check 1 layer only
//
//    // Second linear layer: HIDDEN_FEATURES -> OUT_FEATURES (3072 -> 768)
//    BATCH_LOOP_2: for (int b = 0; b < BATCH_SIZE; b++) {
//        OUT_LOOP_2: for (int j = 0; j < OUT_FEATURES; j++) {
//             float sum = weights2_bias[j];
//            IN_LOOP_2: for (int k = 0; k < HIDDEN_FEATURES; k++) {
//                 sum += hidden_buffer[b][k] * weights2[k * OUT_FEATURES + j];
//            }
//            output_buffer[b * OUT_FEATURES + j] = sum; // Flattened output_buffer
//        }
//    }
//
//    // for residual addition
//    addition(inputs, output_buffer, output_buffer);
//
//    // Store results back to output array
//    OUTPUT_STORE: for (int i = 0; i < BATCH_SIZE; i++) {
//        for (int j = 0; j < OUT_FEATURES; j++) {
//             final[i * OUT_FEATURES + j] = output_buffer[i * OUT_FEATURES + j]; // Use 1D indexing
//        }
//    }
//}
