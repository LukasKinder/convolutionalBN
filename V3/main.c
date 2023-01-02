#include <stdio.h>
#include <math.h>

#include "my_mnist.h"
#include "convolution.h"
#include "bayesianNetwork.h"
#include "convolutionalBayesianNetwork.h"
#include "trainStructure.h"

#define TRAINING_SIZE 60
#define N_NUMBER_NODES 10

int main(void){

    srand(4);

    
    printf("Loading data...\n");
    float *** images = readImagesContinuos("./data/train-images.idx3-ubyte",TRAINING_SIZE);
    printf("finished reading images\n Reading Labels ... \n");
    int * labels = readLabels("./data/train-labels.idx1-ubyte", TRAINING_SIZE);
    printf("Done!\n");

    printf("MAIN: init cbn\n");

    ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2);
    addLayerToCbn(cbn,3,1,1,N_NUMBER_NODES ,1,true);
    addRandomStructure(cbn,0,3);
    tuneCPTwithAugmentedData(cbn,0,"./data/train-images.idx3-ubyte","./data/train-labels.idx1-ubyte",TRAINING_SIZE,0,1.0);
    addLayerToCbn(cbn,5,3,2,N_NUMBER_NODES ,3,false);
    addRandomStructure(cbn,1,3);
    tuneCPTwithAugmentedData(cbn,1,"./data/train-images.idx3-ubyte","./data/train-labels.idx1-ubyte",TRAINING_SIZE,0,1.0);

    printNumberNode(cbn->bayesianNetworks[0]->numberNodes[3],true);
    printNumberNode(cbn->bayesianNetworks[1]->numberNodes[3],true);

    printNode(cbn->bayesianNetworks[0]->nodes[2][24][16],true);
    printNode(cbn->bayesianNetworks[1]->nodes[2][24][16],true);

    saveKernelResponsesOfImage(cbn, images[0]);

    
    free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
}