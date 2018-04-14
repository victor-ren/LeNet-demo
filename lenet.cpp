// CS 533 Fall 2017
// University of Illinois, Urbana-Champaign
// Demo code of LeNet Convolutional Neural Network
// *****************************************
// Before running the code make that the following files are on the SD card
// 1. images.bin
// 2. labels.bin
// 3. params.bin
// This code demos inference on the MNIST dataset with a LeNet CNN
// Provided network parameters have been training already and should give an accuracy of ~98.39%

//#include <c++/6.2.1/iostream>
//#include <c++/6.2.1/cmath>
//#include <c++/6.2.1/algorithm>
//#include <c++/6.2.1/fstream>
//#include <c++/6.2.1/vector>
//#include <c++/6.2.1/array>
//
//#include "xparameters.h"	/* SDK generated parameters */
//#include "xsdps.h"			/* SD device driver */
//#include "xil_printf.h"
//#include "ff.h"				/* FAT File System */
//#include "xil_cache.h"
//#include "xplatform_info.h"
//
//#include "xfc6_hw.h"
//#include "xil_cache_l.h"

// #include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cfloat>

// Max number of test cases in LeNet is 10K
// You can reduce this for testing/debuggin
// Final report must use all 10000 test cases

#define NUM_TESTS 10000
#define XST_FAILURE -1
#define XST_SUCCESS 0

#define IM_SIZE 28
#define NUM_CLASS 10

// conv1: convolutional
#define CONV1_KERNEL_SIZE 5
#define CONV1_NUM_OUT 6
#define CONV1_OUT_SIZE (IM_SIZE - CONV1_KERNEL_SIZE + 1)

// pool1: max pooling
#define MP1_OUT_SIZE (CONV1_OUT_SIZE >> 1) // Divide by two

// conv2: convolutional
#define CONV2_KERNEL_SIZE 5
#define CONV2_NUM_OUT 16
#define CONV2_OUT_SIZE (MP1_OUT_SIZE - CONV2_KERNEL_SIZE + 1)

// pool2: max pooling
#define MP2_OUT_SIZE (CONV2_OUT_SIZE >> 1) // Divide by two

// fc: fully connected / inner product
#define FC_NUM_OUT NUM_CLASS

// SD card and file reading objects
static FILE* fil;		/* File object */
//static FATFS fatfs;
static char *SD_File;
//u32 Platform;

using namespace std;

//Static allocation of test images
unsigned char images[NUM_TESTS * IM_SIZE * IM_SIZE];
unsigned char labels[NUM_TESTS];
// *****************************************

//Static allocation of network parameters and their outputs
float image[1][IM_SIZE][IM_SIZE] = { 0 };
float conv1_weights[CONV1_NUM_OUT][1][CONV1_KERNEL_SIZE][CONV1_KERNEL_SIZE] = { 0 };
float conv1_bias[CONV1_NUM_OUT] = { 0 };
float conv1_output[CONV1_NUM_OUT][CONV1_OUT_SIZE][CONV1_OUT_SIZE] = { 0 };

float pool2_output[CONV1_NUM_OUT][MP1_OUT_SIZE][MP1_OUT_SIZE] = { 0 };

float conv2_weights[CONV2_NUM_OUT][CONV1_NUM_OUT][CONV2_KERNEL_SIZE][CONV2_KERNEL_SIZE] = { 0 };
float conv2_bias[CONV2_NUM_OUT] = { 0 };
float conv2_output[CONV2_NUM_OUT][CONV2_OUT_SIZE][CONV2_OUT_SIZE] = { 0 };

float pool2_output[CONV2_NUM_OUT][MP2_OUT_SIZE][MP2_OUT_SIZE] = { 0 };

float conv5_weights[120][16][5][5] = { 0 };
float conv5_bias[120] = { 0 };
float conv5_output[120][1][1] = { 0 };

float fc6_weights[FC_NUM_OUT][CONV2_NUM_OUT][MP2_OUT_SIZE][MP2_OUT_SIZE] = { 0 };
float fc6_bias[FC_NUM_OUT] = { 0 };
float fc6_output[FC_NUM_OUT] = { 0 };
// *****************************************
// End declaration of layers parameters and buffers
// *****************************************

// Start function definitions of different layers
inline float relu(float input)
{
    return (input > 0) ? input : 0;
}

// Convolution Layer 1
void convolution1(float input[1][IM_SIZE][IM_SIZE], float weights[CONV1_NUM_OUT][1][CONV1_KERNEL_SIZE][CONV1_KERNEL_SIZE], float bias[CONV1_NUM_OUT], float output[CONV1_NUM_OUT][CONV1_OUT_SIZE][CONV1_OUT_SIZE])
{
    // for (int co = 0; co < 6; co++)
    //     for (int h = 0; h < 28; h++)
    //         for (int w = 0; w < 28; w++)
    //         {
    //             float sum = 0;
    //             for (int i = h, m = 0; i < (h + 5); i++, m++)
    //             {
    //                 for (int j = w, n = 0; j < (w + 5); j++, n++)
    //                     sum += weights[co][0][m][n] * input[0][i][j];
    //             }
    //             output[co][h][w] = sum + bias[co];
    //         }
    for (int co = 0; co < CONV1_NUM_OUT; co++)
        for (int h = 0; h < CONV1_OUT_SIZE; h++)
            for (int w = 0; w < CONV1_OUT_SIZE; w++)
            {
                float sum = 0;
                for (int i = h, m = 0; i < (h + CONV1_KERNEL_SIZE); i++, m++)
                {
                    for (int j = w, n = 0; j < (w + CONV1_KERNEL_SIZE); j++, n++)
                        sum += weights[co][0][m][n] * input[0][i][j];
                }
                output[co][h][w] = sum + bias[co];
            }
}

// Relu Layer 1
void relu1(float input[CONV1_NUM_OUT][CONV1_OUT_SIZE][CONV1_OUT_SIZE], float output[CONV1_NUM_OUT][CONV1_OUT_SIZE][CONV1_OUT_SIZE])
{
    for (int i = 0; i < CONV1_NUM_OUT; i++)
        for (int j = 0; j < CONV1_OUT_SIZE; j++)
            for (int k = 0; k < CONV1_OUT_SIZE; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Pooling Layer 2
void max_pooling1(float input[CONV1_NUM_OUT][CONV1_OUT_SIZE][CONV1_OUT_SIZE], float output[CONV1_NUM_OUT][MP1_OUT_SIZE][MP1_OUT_SIZE])
{
    for (int c = 0; c < CONV1_NUM_OUT; c++)
        for (int h = 0; h < MP1_OUT_SIZE; h++)
            for (int w = 0; w < MP1_OUT_SIZE; w++)
            {
                float max_value = -FLT_MAX;
                for (int i = h * 2; i < h * 2 + 2; i++)
                {
                    for (int j = w * 2; j < w * 2 + 2; j++)
                        max_value = (max_value > input[c][i][j]) ? max_value : input[c][i][j];
                }
                output[c][h][w] = max_value;

            }
}

// Relu Layer 2
void relu2(float input[CONV2_NUM_OUT][CONV2_OUT_SIZE][CONV2_OUT_SIZE], float output[CONV2_NUM_OUT][CONV2_OUT_SIZE][CONV2_OUT_SIZE])
{
    for (int i = 0; i < CONV2_NUM_OUT; i++)
        for (int j = 0; j < CONV2_OUT_SIZE; j++)
            for (int k = 0; k < CONV2_OUT_SIZE; k++)
                output[i][j][k] = relu(input[i][j][k]);
}

// Convolution Layer 3
void convolution2(float input[CONV1_NUM_OUT][MP1_OUT_SIZE][MP1_OUT_SIZE], float weights[CONV2_NUM_OUT][CONV1_NUM_OUT][CONV2_KERNEL_SIZE][CONV2_KERNEL_SIZE], float bias[CONV2_NUM_OUT], float output[CONV2_NUM_OUT][CONV2_OUT_SIZE][CONV2_OUT_SIZE])
{
    for (int co = 0; co < CONV2_NUM_OUT; co++){
        for (int h = 0; h < CONV2_OUT_SIZE; h++){
            for (int w = 0; w < CONV2_OUT_SIZE; w++){
                float sum = 0;
                for (int i = h, m = 0; i < (h + CONV2_KERNEL_SIZE); i++, m++){
                    for (int j = w, n = 0; j < (w + CONV2_KERNEL_SIZE); j++, n++){
                        for (int ci = 0; ci < CONV1_NUM_OUT; ci++){
                            sum += weights[co][ci][m][n] * input[ci][i][j];
                        }
                    }
                }
                output[co][h][w] = sum + bias[co];
            }
        }
    }
}

// Pooling Layer 4
void max_pooling2(float input[CONV2_NUM_OUT][CONV2_OUT_SIZE][CONV2_OUT_SIZE], float output[CONV2_NUM_OUT][MP2_OUT_SIZE][MP2_OUT_SIZE])
{
    for (int c = 0; c < CONV2_NUM_OUT; c++){
        for (int h = 0; h < CONV2_OUT_SIZE; h++){
            for (int w = 0; w < CONV2_OUT_SIZE; w++){
                float max_value = -FLT_MAX;
                for (int i = h * 2; i < h * 2 + 2; i++){
                    for (int j = w * 2; j < w * 2 + 2; j++){
                        max_value = (max_value > input[c][i][j]) ? max_value : input[c][i][j];
                    }
                }
                output[c][h][w] = max_value;
            }
        }
    }
}

// Fully connected Layer 6
void fc(const float input[CONV2_NUM_OUT][MP2_OUT_SIZE][MP2_OUT_SIZE], const float weights[FC_NUM_OUT][CONV2_NUM_OUT][MP2_OUT_SIZE][MP2_OUT_SIZE], const float bias[FC_NUM_OUT], float output[FC_NUM_OUT])
{
        for(int n = 0; n < FC_NUM_OUT; n++){
            output[n] = 0;
            for(int c = 0; c < 16; c++){
                for(int y = 0; y < MP2_OUT_SIZE; y++){
                    for (int x = 0; x < MP2_OUT_SIZE; x++){
                        output[n] += weights[n][c][y][x] * input[c][y][x];
                    }
                }
            }
            output[n]+=bias[n];
        }
}


// *****************************************
// End declaration of layers functions
// *****************************************

// Parse MNIST test images
int parse_mnist_images(const char* filename, unsigned char *images)
{
    int Res;
    int NumBytesRead;
    SD_File = (char *)filename;
    unsigned int header[4];

    fil = fopen(SD_File, "rb");
    if (!fil)
    {
        printf("ERROR when opening mnist images data file!\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Opened mnist images data file\n\r");
    }

    Res = fseek(fil, 0, SEEK_SET);
    if (Res)
    {
        printf("Cant seek to start\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Seeked to start\n\r");
    }

    Res = fread((void*)header, sizeof(unsigned int), 4, fil);
    if (Res == 0)
    {
        printf("Cant read header from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read header from file\n\r");
    }

    NumBytesRead = fread((void*)images, sizeof(unsigned char), NUM_TESTS * 28 * 28, fil);
    if (NumBytesRead == 0)
    {
        printf("Cant read images from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read images from file\n\r");
    }

    Res = fclose(fil);
    if (Res)
    {
        printf("Failed to close images file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Closed images file\n\r");
    }

    printf("Returning...\n\r");


    return XST_SUCCESS;
}

// Parse MNIST test image labels
int parse_mnist_labels(const char* filename, unsigned char *labels)
{
    int Res;
    int NumBytesRead;
    SD_File = (char *)filename;
    unsigned int header[2];

    fil = fopen(SD_File, "rb");
    if (!fil)
    {
        printf("ERROR when opening mnist label data file!\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Opened mnist labels data file\n\r");
    }

    Res = fseek(fil, 0, SEEK_SET);
    if (Res)
    {
        printf("Cant seek to start\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Seeked to start\n\r");
    }

    NumBytesRead = fread((void*)header, sizeof(unsigned int), 2, fil);
    if (NumBytesRead == 0)
    {
        printf("Cant read header from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read header from file\n\r");
    }

    NumBytesRead = fread((void*)labels, sizeof(unsigned char), NUM_TESTS, fil);
    if (NumBytesRead == 0)
    {
        printf("Cant read labels from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read labels from file\n\r");
    }

    Res = fclose(fil);
    if (Res)
    {
        printf("Failed to close labels file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Closed labels file\n\r");
    }

    return 0;


}

// Parse parameter file and load it in to the arrays
int parse_parameters()
{
    int Res;
    int NumBytesRead;

    fil = fopen("200_iter/_conv1_200.bin", "rb");
    if (!fil)
    {
        printf("ERROR when opening parameter file (_conv1_200)!\n\r");
        return XST_FAILURE;
    }

    NumBytesRead = fread((void*)conv1_weights, sizeof(float), 6*5*5, fil);
    if (NumBytesRead == 0)
    {
        printf("Cant read conv1_weights from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read conv1_weights (%d bytes) from file\n\r", NumBytesRead);
    }
    fclose(fil);

    fil = fopen("200_iter/bias_conv1_200.bin", "rb");
    if (!fil){
        printf("ERROR when opening parameter file (bias_conv1_200)!\n\r");
        return XST_FAILURE;
    }
    NumBytesRead = fread((void*)conv1_bias, sizeof(float), 6, fil);
    if (NumBytesRead == 0)
    {
        printf("Cant read conv1_bias from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read conv1_bias from file\n\r");
    }
    fclose(fil);

    fil = fopen("200_iter/_conv2_200.bin", "rb");
    if (!fil){
        printf("ERROR when opening parameter file (_conv2_200)!\n\r");
        return XST_FAILURE;
    }
    NumBytesRead = fread((void*)conv3_weights, sizeof(float), 16*6*5*5, fil);
    if (NumBytesRead == 0)
    {
        printf("Cant read conv2_weights from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read conv3_weights from file\n\r");
    }
    fclose(fil);

    fil = fopen("200_iter/bias_conv2_200.bin", "rb");
    if (!fil){
        printf("ERROR when opening parameter file (bias_conv2_200)!\n\r");
        return XST_FAILURE;
    }
    NumBytesRead = fread((void*)conv3_bias, sizeof(float), 16, fil);
    if (NumBytesRead == 0)
    {
        printf("Cant read conv3_bias from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read conv3_bias from file\n\r");
    }
    fclose(fil);

    fil = fopen("200_iter/_score_200.bin", "rb");
    if (!fil){
        printf("ERROR when opening parameter file (_score_200)!\n\r");
        return XST_FAILURE;
    }
    NumBytesRead = fread((void*)fc6_weights, sizeof(float), 10*16*4*4, fil);
    if (NumBytesRead == 0)
    {
        printf("Cant read fc6_weights from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read fc6_weights from file\n\r");
    }
    fclose(fil);

    fil = fopen("200_iter/bias_score_200.bin", "rb");
    if (!fil){
        printf("ERROR when opening parameter file (bias_score_200)!\n\r");
        return XST_FAILURE;
    }
    NumBytesRead = fread((void*)fc6_bias, sizeof(float), 10, fil);
    if (NumBytesRead == 0)
    {
        printf("Cant read fc6_bias from file\n\r");
        return XST_FAILURE;
    }
    else
    {
        printf("Read fc6_bias from file\n\r");
    }
    fclose(fil);

    return 0;

}

// Fetch a single image to be processed.
//
void get_image(unsigned char *images, unsigned int idx, float image[1][32][32])
{
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
        {
            if (i < 2 || i > 29 || j < 2 || j > 29)
                image[0][i][j] = -1.0;
            else
                image[0][i][j] = images[idx * 28 * 28 + (i - 2) * 28 + j - 2] / (float)255.0 * (float)2.0 - (float)1.0; // Linear scale
        }
}

int main(int argc, char **argv)
{
    //cout << "Starting LeNet\n\r";

    //cout << "Mounting SD\n\r";
    //FRESULT Res;
    //TCHAR *Path = "0:/";
    //Res = f_mount(&fatfs, Path, 0);
    //if (Res != FR_OK)
    //{
    //    xil_printf("Failed to open filesystem\n\r");
    //    return XST_FAILURE;
    //}
    //else
    //{
    //    xil_printf("Mounted card\n\r");
    //}

    //cout<<"Creating test data matrices\n\r";
    //cout<<"Creating layer matrices\n\r";


    cout << "Parsing MNIST images\n\r";
    parse_mnist_images("images.bin", images);
    //xil_printf("Back from image load\n\r");

    cout << "Parsing MNIST labels\n\r";
    parse_mnist_labels("labels.bin", labels);

    cout << "Parsing parameters\n\r";
    parse_parameters();

    cout << "Starting inference\n\r";
    int num_correct = 0;

    printf("\n\rTest Image: ");
    for (int k = 0; k < NUM_TESTS; k++)
    {
        //Get test image from dataset
        get_image(images, k, image);

        //Begin inference here.
        convolution1(image, conv1_weights, conv1_bias, conv1_output);
        relu1(conv1_output, conv1_output);

        max_pooling2(conv1_output, pool2_output);
        relu2(pool2_output, pool2_output);

        convolution3(pool2_output, conv3_weights, conv3_bias, conv3_output);
        relu3(conv3_output, conv3_output);

        max_pooling4(conv3_output, pool4_output);
        relu4(pool4_output, pool4_output);

        convolution5(pool4_output, conv5_weights, conv5_bias, conv5_output);
        relu5(conv5_output, conv5_output);

        fc6(conv5_output, fc6_weights, fc6_bias, fc6_output);
        //End inference here.

        //Check which output was largest.
        unsigned char result = 0;
        float p = -FLT_MAX;
        for (int i = 0; i < 10; i++)
        {
            if (fc6_output[i] > p)
            {
                p = fc6_output[i];
                result = i;
            }
        }
        //Largest output is result

        //std::cout << "test " << k << ": " << int(result) << "/" << int(labels[k]) << ": ";
        if (result == labels[k])
        {
            num_correct++;
            //std::cout << "correct" << std::endl;
        }
        else
        {
            //std::cout << "WRONG" << std::endl;
        }

        //Disable these print statements when doing profiling and benchmark times
        printf("%d ", k);
        if (k % 10 == 0)
            printf("\n\rTest Image: ");
    }

    std::cout << "\n\rAccuracy = " << float(num_correct) / NUM_TESTS * 100.0 << "%" << std::endl;

    return 0;
}
