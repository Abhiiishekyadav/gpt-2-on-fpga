
#include "mhanew.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void mha(
    float inputs[5 * 768],      // Input shape: 5x768
    float weights[768 * 2304],  // Weight shape: 768x2304
    float bias[2304],           // Bias shape: 2304
	float dense_weights[768*768],
	float dense_bias[768],
    float final[5 * 2304]       // Output shape: 5x2304
);
int main() {
    // Define constants for matrix dimensions
    const int input_size = 5;
    const int input_dim = 768;
    const int output_dim = 2304;
    const int dense_dim = 768;

    // Calculate total sizes
    const int input_total_size = input_size * input_dim;
    const int weight_total_size = input_dim * output_dim;
    const int dense_weight_total_size = dense_dim * dense_dim;
    const int output_total_size = input_size * output_dim;

    // Allocate memory
    float* inputss = (float*)malloc(input_total_size * sizeof(float));
    float* weights = (float*)malloc(weight_total_size * sizeof(float));
    float* bias = (float*)malloc(output_dim * sizeof(float));
    float* dense_weights = (float*)malloc(dense_weight_total_size * sizeof(float));
    float* dense_bias = (float*)malloc(dense_dim * sizeof(float));
    float* output = (float*)malloc(output_total_size * sizeof(float));

    // Check memory allocation
    if (!inputss || !weights || !bias || !dense_weights || !dense_bias || !output) {
        printf("Error: Memory allocation failed!\n");
        return -1;
    }
    // Copy data from header file arrays
    printf("Copying input data...\n");
    for (int i = 0; i < input_total_size; i++) {
        inputss[i] = inputs[i];  // Copy from external array
    }

    printf("Copying weights...\n");
    for (int i = 0; i < weight_total_size; i++) {
        weights[i] = weights1[i];  // Copy from external array
    }

    printf("Copying bias...\n");
    for (int i = 0; i < output_dim; i++) {
        bias[i] = bias1[i];  // Copy from external array
    }
    printf("Initializing dense layer weights and bias...\n");
         for (int i = 0; i < dense_weight_total_size; i++) {
            dense_weights[i] =  weights2[i];
        }

         for (int i = 0; i < dense_dim; i++) {
            dense_bias[i] =bias2[i];
        }
    // Verify data copying
    printf("\nVerifying copied data:\n");
    printf("First few input values:\n");
    for (int i = 0; i < 5; i++) {
        printf("inputss[%d] = %f, original inputs[%d] = %f\n",
               i, inputss[i], i, inputs[i]);
    }

    printf("\nFirst few weight values:\n");
    for (int i = 0; i < 5; i++) {
        printf("weights[%d] = %f, original weights1[%d] = %f\n",
               i, weights[i], i, weights1[i]);
    }

    printf("\nFirst few bias values:\n");
    for (int i = 0; i < 5; i++) {
        printf("bias[%d] = %f, original bias1[%d] = %f\n",
               i, bias[i], i, bias1[i]);
    }

    printf("\nFirst few dense weight values:\n");
       for (int i = 0; i < 5; i++) {
           printf("dense_weights[%d] = %f\n", i, dense_weights[i]);
       }
       printf("\nFirst few dense bias values:\n");
          for (int i = 0; i < 5; i++) {
              printf("bias2[%d] = %f\n", i, dense_bias[i]);
          }


    // Run MHA
    printf("\nRunning MHA implementation...\n");
    mha(inputss, weights, bias, dense_weights, dense_bias, output);
    // Print results
    printf("\nMHA Output (first 10 values):\n");
    for (int i = 0; i < 500 && i < output_total_size; i++) {
        printf("output[%d] = %f\n", i, output[i]);
    }

    // Calculate statistics
    float min_val = output[0];
    float max_val = output[0];
    float sum = 0.0f;

    for (int i = 0; i < output_total_size; i++) {
        if (output[i] < min_val) min_val = output[i];
        if (output[i] > max_val) max_val = output[i];
        sum += output[i];
    }

    printf("\nOutput Statistics:\n");
    printf("Minimum value: %f\n", min_val);
    printf("Maximum value: %f\n", max_val);
    printf("Average value: %f\n", sum / output_total_size);

    free(inputss);
    free(weights);
    free(bias);
    free(dense_weights);
    free(dense_bias);
    free(output);
    printf("\nTest completed successfully!\n");
    return 0;
}
//#include "mha.h"
//#include <stdio.h>
//#include <math.h>
//
//// Define constants for matrix dimensions
//#define INPUT_SIZE 5
//#define INPUT_DIM 768
//#define OUTPUT_DIM 2304
//
//// Calculate total sizes
//#define INPUT_TOTAL_SIZE (INPUT_SIZE * INPUT_DIM)
//#define WEIGHT_TOTAL_SIZE (INPUT_DIM * OUTPUT_DIM)
//#define OUTPUT_TOTAL_SIZE (INPUT_SIZE * OUTPUT_DIM)
//
//int main() {
//    // Declare static arrays
//    static float inputss[INPUT_TOTAL_SIZE];
//    static float weights[WEIGHT_TOTAL_SIZE];
//    static float bias[OUTPUT_DIM];
//    static float output[OUTPUT_TOTAL_SIZE];
//
//    // Copy data from header file arrays
//    printf("Copying input data...\n");
//    for (int i = 0; i < INPUT_TOTAL_SIZE; i++) {
//        inputss[i] = inputs[i];  // Copy from external array
//    }
//
//    printf("Copying weights...\n");
//    for (int i = 0; i < WEIGHT_TOTAL_SIZE; i++) {
//        weights[i] = weights1[i];  // Copy from external array
//    }
//
//    printf("Copying bias...\n");
//    for (int i = 0; i < OUTPUT_DIM; i++) {
//        bias[i] = bias1[i];  // Copy from external array
//    }
//
//    // Verify data copying
//    printf("\nVerifying copied data:\n");
//    printf("First few input values:\n");
//    for (int i = 0; i < 5; i++) {
//        printf("inputss[%d] = %f, original inputs[%d] = %f\n",
//               i, inputss[i], i, inputs[i]);
//    }
//
//    printf("\nFirst few weight values:\n");
//    for (int i = 0; i < 5; i++) {
//        printf("weights[%d] = %f, original weights1[%d] = %f\n",
//               i, weights[i], i, weights1[i]);
//    }
//
//    printf("\nFirst few bias values:\n");
//    for (int i = 0; i < 5; i++) {
//        printf("bias[%d] = %f, original bias1[%d] = %f\n",
//               i, bias[i], i, bias1[i]);
//    }
//
//    // Run MHA
//    printf("\nRunning MHA implementation...\n");
//    mha(inputss, weights, bias, output);
//
//    // Print results
//    printf("\nMHA Output (first 10 values):\n");
//    for (int i = 0; i < 10 && i < OUTPUT_TOTAL_SIZE; i++) {
//        printf("output[%d] = %f\n", i, output[i]);
//    }
//
//    // Calculate statistics
//    float min_val = output[0];
//    float max_val = output[0];
//    float sum = 0.0f;
//
//    for (int i = 0; i < OUTPUT_TOTAL_SIZE; i++) {
//        if (output[i] < min_val) min_val = output[i];
//        if (output[i] > max_val) max_val = output[i];
//        sum += output[i];
//    }
//
//    printf("\nOutput Statistics:\n");
//    printf("Minimum value: %f\n", min_val);
//    printf("Maximum value: %f\n", max_val);
//    printf("Average value: %f\n", sum / OUTPUT_TOTAL_SIZE);
//
//    printf("\nTest completed successfully!\n");
//    return 0;
//}
