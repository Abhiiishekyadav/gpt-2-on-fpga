#include "add.h"
#include <stdio.h>
#include <float.h>
#include <math.h>


void tokenizer(float wpe[10 * 768], float wte[10 * 768], float result[10 * 768]);




int main()
{
    float result[10 * 768];
    float l_wpe[10 * 768];
    float l_wte[10 * 768];
    for (int i = 0; i < 3840; i++) {
    	l_wpe[i] = (float)wpe[i];
    	l_wte[i]= (float)wte[i];
       }




    tokenizer(l_wpe, l_wte, result);

    printf("\nFinal output array (result):\n");
    for (int i = 0; i < 10; i++) {  // Print first 10 elements of result
        if (isnan(result[i]) || isinf(result[i]) || fabs(result[i]) > FLT_MAX) {
            printf("Error: result[%d] = %f (Invalid value)\n", i, result[i]);
        } else {
            printf("result[%d] = %f\n", i, result[i]);
        }
    }

     printf("\nDebug: Final result array (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("result[%d] = %f\n", i, result[i]);
    }

    printf("Execution completed\n");
    return 0;
}
