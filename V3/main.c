#include <stdio.h>
#include <math.h>

#include "my_mnist.h"
#include "convolution.h"
#include "bayesianNetwork.h"
#include "convolutionalBayesianNetwork.h"
#include "trainStructure.h"
#include "trainKernels.h"
#include "performance_measure.h"

#define TRAINING_SIZE 600
#define N_NUMBER_NODES 10

#define TRAIN_IMAGE_PATH "./data/train-images.idx3-ubyte"
#define TRAIN_Label_PATH "./data/train-labels.idx1-ubyte"

#define TEST_IMAGE_PATH "./data/t10k-images.idx3-ubyte"
#define TEST_Label_PATH "./data/t10k-labels.idx1-ubyte"

int main(void){

    srand(5);

    
    printf("Loading data...\n");
    float *** images = readImagesContinuos(TRAIN_IMAGE_PATH,TRAINING_SIZE);
    printf("finished reading images\n Reading Labels ... \n");
    int * labels = readLabels(TRAIN_Label_PATH, TRAINING_SIZE);
    printf("Done!\n");

    printf("MAIN: init cbn\n");
    ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2);


    //layer 1
    addLayerToCbn(cbn,3,1,1,N_NUMBER_NODES ,1,true);
    initKernels(cbn,0,images,10,false);
    optimizeStructure(cbn,0,3,images,labels,TRAINING_SIZE,true);

    tuneCPTwithAugmentedData(cbn,0,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,TRAINING_SIZE,0,1.0);

    printNumberNode(cbn->bayesianNetworks[0]->numberNodes[0],true);
    printNode(cbn->bayesianNetworks[0]->nodes[0][24][16],true);

    //layer 2
    addLayerToCbn(cbn,5,3,2,N_NUMBER_NODES ,3,false);
    initKernels(cbn,1,images,10,false);
    optimizeStructure(cbn,1,3,images,labels,TRAINING_SIZE,true);
    tuneCPTwithAugmentedData(cbn,1,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,TRAINING_SIZE,0,1.0);

    printNumberNode(cbn->bayesianNetworks[1]->numberNodes[3],true);
    printNode(cbn->bayesianNetworks[1]->nodes[2][24][16],true);



    saveKernelResponsesOfImage(cbn, images[0]);
    printf("Accuracy = %f\n",predictNumberAccuracy(cbn,TEST_IMAGE_PATH,TEST_Label_PATH,100,true));

    
    free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
}