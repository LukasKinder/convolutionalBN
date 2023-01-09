#include <stdio.h>
#include <math.h>

#include "my_mnist.h"
#include "convolution.h"
#include "bayesianNetwork.h"
#include "convolutionalBayesianNetwork.h"
#include "trainStructure.h"
#include "trainKernels.h"
#include "performance_measure.h"

#define TRAINING_SIZE 60000
#define N_NUMBER_NODES 20
#define N_INCOMING_RELATIONS 5

#define TRAIN_IMAGE_PATH "./data/train-images.idx3-ubyte"
#define TRAIN_Label_PATH "./data/train-labels.idx1-ubyte"

#define TEST_IMAGE_PATH "./data/t10k-images.idx3-ubyte"
#define TEST_Label_PATH "./data/t10k-labels.idx1-ubyte"

#define KERNEL_SEARCH_ITERATIONS 16

void hardcodeKernels(Kernel * kernels){

    //N-S edge
    kernels[0].weights[0][0][0] = -1; kernels[0].weights[0][1][0] = -1.0;
    kernels[0].weights[0][0][1] = 1; kernels[0].weights[0][1][1] = 1.0;
    kernels[0].bias = -0.8;

    //E-W edge
    kernels[1].weights[0][0][0] = 1; kernels[1].weights[0][1][0] = -1.0;
    kernels[1].weights[0][0][1] = 1; kernels[1].weights[0][1][1] = -1.0;
    kernels[1].bias = -0.8;

    //S-N edge
    kernels[2].weights[0][0][0] = 1.0; kernels[2].weights[0][1][0] = 1.0;
    kernels[2].weights[0][0][1] = -1.0; kernels[2].weights[0][1][1] = -1.0;
    kernels[2].bias = -0.8;

    //W-E edge
    kernels[3].weights[0][0][0] = -1.0; kernels[3].weights[0][1][0] = 1.0;
    kernels[3].weights[0][0][1] = -1.0; kernels[3].weights[0][1][1] = 1.0;
    kernels[3].bias = -0.8;

    //full black
    kernels[4].weights[0][0][0] = -1.0; kernels[4].weights[0][1][0] = -1.0;
    kernels[4].weights[0][0][1] = -1.0; kernels[4].weights[0][1][1] = -1.0;
    kernels[4].bias = 0.4;

    //full white
    kernels[5].weights[0][0][0] = 1.0; kernels[5].weights[0][1][0] = 1.0;
    kernels[5].weights[0][0][1] = 1.0; kernels[5].weights[0][1][1] = 1.0;
    kernels[5].bias = -3.5;

    //up-right corner
    kernels[6].weights[0][0][0] = -1.0; kernels[6].weights[0][1][0] = 1.0;
    kernels[6].weights[0][0][1] = -1.0; kernels[6].weights[0][1][1] = -1.0;
    kernels[6].bias = -0.3;

    //up-left corner
    kernels[7].weights[0][0][0] = 1.0; kernels[7].weights[0][1][0] = -1.0;
    kernels[7].weights[0][0][1] = -1.0; kernels[7].weights[0][1][1] = -1.0;
    kernels[7].bias = -0.3;

    //down-right corner
    kernels[8].weights[0][0][0] = -1.0; kernels[8].weights[0][1][0] = -1.0;
    kernels[8].weights[0][0][1] = -1.0; kernels[8].weights[0][1][1] = 1.0;
    kernels[8].bias = -0.3;

    //down-left corner
    kernels[9].weights[0][0][0] = -1.0; kernels[9].weights[0][1][0] = -1.0;
    kernels[9].weights[0][0][1] = 1.0; kernels[9].weights[0][1][1] = -1.0;
    kernels[9].bias = -0.3;
} 

int main(void){

    srand(8);

    float *** images = readImagesContinuos(TRAIN_IMAGE_PATH,TRAINING_SIZE);
    int * labels = readLabels(TRAIN_Label_PATH, TRAINING_SIZE);

    ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(3);

    //layer 1
    addLayerToCbn(cbn,10,2,2,0,2,true);
    addLayerToCbn(cbn,7,3,2,N_NUMBER_NODES ,5,true);
    addLayerToCbn(cbn,10,3,2,N_NUMBER_NODES ,8,true);


    initKernels(cbn,0,images,10,false);
    hardcodeKernels(cbn->transitionalKernels[0]);
    initKernels(cbn,1,images,10,false);
    initKernels(cbn,2,images,10,false);


    saveKernelResponsesOfImageExperimental(cbn, images[0]);

    kernelTrainingWhileUpdatingStructure(cbn,0,0,10,6.0,0.5,2000,images,labels,TRAINING_SIZE,N_INCOMING_RELATIONS,5000,true);
    kernelTrainingWhileUpdatingStructure(cbn,1,KERNEL_SEARCH_ITERATIONS,8,0.005,0.5,3000,images,labels,TRAINING_SIZE,N_INCOMING_RELATIONS,3000,true);
    kernelTrainingWhileUpdatingStructure(cbn,2,KERNEL_SEARCH_ITERATIONS,8,0.005,0.5,3000,images,labels,TRAINING_SIZE,N_INCOMING_RELATIONS,3000,true);

    /* for (int i = 0; i < cbn->n_kernels[1]; i++){
        printKernel(cbn->transitionalKernels[1][i]);
    } */

    printNumberNode(cbn->bayesianNetworks[1]->numberNodes[0],true);
    printNumberNode(cbn->bayesianNetworks[2]->numberNodes[0],true);


    saveKernelResponsesOfImage(cbn, images[0]);

    tuneCPTwithAugmentedData(cbn,0,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,TRAINING_SIZE,0,1.0);
    tuneCPTwithAugmentedData(cbn,1,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,TRAINING_SIZE,0,1.0);
    tuneCPTwithAugmentedData(cbn,2,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,TRAINING_SIZE,0,1.0);

    printf("Accuracy = %.2f\n",predictNumberAccuracy(cbn,TEST_IMAGE_PATH,TEST_Label_PATH,1000,true));

    
    free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
}