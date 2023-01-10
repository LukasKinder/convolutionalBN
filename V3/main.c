#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "my_mnist.h"
#include "convolution.h"
#include "bayesianNetwork.h"
#include "convolutionalBayesianNetwork.h"
#include "trainStructure.h"
#include "trainKernels.h"
#include "performance_measure.h"

#define TRAINING_SIZE 60000
#define N_NUMBER_NODES 50
#define N_INCOMING_RELATIONS 12

#define TRAIN_IMAGE_PATH "./data/train-images.idx3-ubyte"
#define TRAIN_Label_PATH "./data/train-labels.idx1-ubyte"

#define TEST_IMAGE_PATH "./data/t10k-images.idx3-ubyte"
#define TEST_Label_PATH "./data/t10k-labels.idx1-ubyte"

#define KERNEL_SEARCH_ITERATIONS 10

int main(void){

    srand(8);

    float *** images = readImagesContinuos(TRAIN_IMAGE_PATH,TRAINING_SIZE);
    int * labels = readLabels(TRAIN_Label_PATH, TRAINING_SIZE);

    ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2);

    //layer 1
    addLayerToCbn(cbn,10,2,2, N_NUMBER_NODES ,2,true);
    addLayerToCbn(cbn,10,3,2, N_NUMBER_NODES,5,true);

    loadKernels(cbn->transitionalKernels[0], cbn->n_kernels[0],"hardcoded_layer1_n10_d1_s2");
    loadKernels(cbn->transitionalKernels[1], cbn->n_kernels[1],"layer2_n10_d10_s3");

    //saveKernelResponsesOfImageExperimental(cbn, images[0]);

    //kernelTrainingWhileUpdatingStructure(cbn,1,KERNEL_SEARCH_ITERATIONS,10,0.005,0.5,100,images,labels,TRAINING_SIZE,N_INCOMING_RELATIONS,2000,true);
    //saveKernels(cbn->transitionalKernels[1],10,"layer2_n10_d10_s3");

    printf("optimize structure layer 0\n");
    optimizeStructure(cbn,0,N_INCOMING_RELATIONS,images,labels,TRAINING_SIZE,10,0.005,0,false); 
    printf("optimize structure layer 1\n");
    optimizeStructure(cbn,1,N_INCOMING_RELATIONS,images,labels,TRAINING_SIZE,10,0.005,0,true); 


    //saveKernelResponsesOfImage(cbn, images[0]);

    tuneCPTwithAugmentedData(cbn,0,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,60000,0,1.0);
    tuneCPTwithAugmentedData(cbn,1,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,60000,0,1.0);

    printf("Accuracy = %.3f\n",predictNumberAccuracy(cbn,TEST_IMAGE_PATH,TEST_Label_PATH,2000,true));

    
    free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
}