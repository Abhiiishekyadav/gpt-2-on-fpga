//#include "norm.h"
//#include "mlp.h"
//#include <stdio.h>
//#include <float.h>
//#include <math.h>
//
//#define BATCH_SIZE 5
//#define IN_FEATURES 768
//#define HIDDEN_FEATURES 3072
//#define OUT_FEATURES 768
//#define PI 3.14159265358979323846f
//#define SQRT_2_DIV_PI 0.797884560802865f
//void ffn(
//    float inputs[BATCH_SIZE * IN_FEATURES],    // Input: (BATCH_SIZE, IN_FEATURES)
//    float ln2_b[IN_FEATURES],                  // Bias for layer normalization
//    float ln2_g[IN_FEATURES],                  // Gain for layer normalization
//	float bias[3072],
// 	    float weights[768 * 256],
//    float final[BATCH_SIZE * OUT_FEATURES]     // Output: (BATCH_SIZE, OUT_FEATURES)
//);
//
//int main() {
//
//    float inputs[BATCH_SIZE * IN_FEATURES];
//    float localln2_b[IN_FEATURES];
//    float localln2_g[IN_FEATURES];
//    float final[BATCH_SIZE * IN_FEATURES] ;
//	float local_bias[3072] ;
//		float  local_weights[768 * 256] ;
//
//     for (int i = 0; i < IN_FEATURES; i++) {
//         localln2_b[i] = ln2_b[i];  // Copy values from ln2_b to localln2_b
//         localln2_g[i] = ln2_g[i];  // Copy values from ln2_g to localln2_g
//     }
//
//      for (int i = 0; i < BATCH_SIZE * IN_FEATURES; i++) {
//         inputs[i] = input[i];
//         //printf("Debug: inputs[%d] = %f\n", i, inputs[i]);
//     }
//
//      for (int i = 0; i < 3072; i++) {
//    	  local_bias[i] = mlp_bias[i];
//    	 // printf("Debug: local_bias[%d] = %f\n", i, local_bias[i]);
//      }
//
//      for (int i = 0; i < 235; i++) {
//    	  local_weights[i] = mlp_w[i];
//      	 // printf("Debug: local_weights[%d] = %f\n", i, local_weights[i]);
//      }
//
//
//    // Call the ffn function
//    ffn(inputs, localln2_b, localln2_g,local_bias, local_weights, final);
//
//    // Print the final output for validation
//    printf("\nFinal output array (all 768 elements):\n");
//    for (int i = 0; i < 100; i++) {
//        if (isnan(final[i]) || isinf(final[i]) || fabs(final[i]) > FLT_MAX) {
//            printf("Error: final[%d] = %f (Invalid value)\n", i, final[i]);
//        } else {
//            printf("final[%d] = %f\n", i, final[i]);
//        }
//    }
//
//    printf("Execution completed\n");
//    return 0;
//}
#include "mlp.h"
#include "norm.h"
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>  // For malloc and free

#define BATCH_SIZE 5
#define IN_FEATURES 768
#define HIDDEN_FEATURES 3072
#define OUT_FEATURES 768
#define PI 3.14159265358979323846f
#define SQRT_2_DIV_PI 0.797884560802865f

void ffn(
    float* inputs,     // Dynamically allocated memory for (BATCH_SIZE * IN_FEATURES)
    float* ln2_b,      // Dynamically allocated memory for (IN_FEATURES)
    float* ln2_g,      // Dynamically allocated memory for (IN_FEATURES)
    float* bias,       // Dynamically allocated memory for (3072)
    float* weights,    // Dynamically allocated memory for (768 * 256)
    float* final       // Dynamically allocated memory for (BATCH_SIZE * OUT_FEATURES)
);

int main() {
    // Dynamically allocate memory for the arrays
    float* inputs = (float*)malloc(BATCH_SIZE * IN_FEATURES * sizeof(float));
    float* localln2_b = (float*)malloc(IN_FEATURES * sizeof(float));
    float* localln2_g = (float*)malloc(IN_FEATURES * sizeof(float));
    float* final = (float*)malloc(BATCH_SIZE * OUT_FEATURES * sizeof(float));
    float* local_bias = (float*)malloc(3072 * sizeof(float));
    float* local_weights = (float*)malloc(768 * 256 * sizeof(float));

    // Check if the memory allocation was successful
    if (!inputs || !localln2_b || !localln2_g || !final || !local_bias || !local_weights) {
        printf("Error: Memory allocation failed!\n");
        return -1;  // Exit if memory allocation fails
    }
    // Debug: Print inputs initialization
      printf("Initializing inputs...\n");
      for (int i = 0; i <3840; i++) {
          inputs[i] = input[i];  // Assuming 'input' is already initialized
          // Debug print to check values
       }
    // Initialize values (assuming 'ln2_b', 'ln2_g', 'input', 'mlp_bias', and 'mlp_w' are defined elsewhere)
    for (int i = 0; i < IN_FEATURES; i++) {
        localln2_b[i] = ln2_b[i];  // Copy values from ln2_b to localln2_b
        localln2_g[i] = ln2_g[i];  // Copy values from ln2_g to localln2_g
    }



    // Debug: Print biases and weights initialization
    printf("Initializing local_bias...\n");
    for (int i = 0; i < 3072; i++) {
        local_bias[i] = mlp_bias[i];  // Assuming 'mlp_bias' is already initialized
    }

    printf("Initializing local_weights...\n");
    for (int i = 0; i < 768 * 256; i++) {
        local_weights[i] = mlp_w[i];  // Assuming 'mlp_w' is already initialized
    }

    // Call the ffn function
    printf("Calling ffn function...\n");
    ffn(inputs, localln2_b, localln2_g, local_bias, local_weights, final);

    // Print the final output for validation
    printf("\nFinal output array (first 100 elements):\n");
    for (int i = 0; i < 100; i++) {
        if (isnan(final[i]) || isinf(final[i]) || fabs(final[i]) > FLT_MAX) {
            printf("Error: final[%d] = %f (Invalid value)\n", i, final[i]);
        } else {
            printf("final[%d] = %f\n", i, final[i]);
        }
    }

    // Free dynamically allocated memory
    free(inputs);
    free(localln2_b);
    free(localln2_g);
    free(final);
    free(local_bias);
    free(local_weights);

    printf("Execution completed\n");

    return 0;
}



