#include <stdio.h>
#include <math.h>

#include "my_mnist.h"
#include "convolution.h"
#include "bayesianNetwork.h"
#include "convolutionalBayesianNetwork.h"

#define TRAINING_SIZE 60

int main(void){

    srand(2);

    
    printf("Loading data...\n");
    float *** images = readImagesContinuos("./data/train-images.idx3-ubyte",TRAINING_SIZE);
    printf("finished reading images\n Reading Labels ... \n");
    int * labels = readLabels("./data/train-labels.idx1-ubyte", TRAINING_SIZE);
    printf("Done!\n");

    printf("MAIN: init cbn\n");

    ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2);
    addLayerToCbn(cbn,3,1,1,1,true);
    tuneCPTwithAugmentedData(cbn,0,"./data/train-images.idx3-ubyte","./data/train-labels.idx1-ubyte",TRAINING_SIZE,1,1.0);
    addLayerToCbn(cbn,5,3,2,3,false);
    tuneCPTwithAugmentedData(cbn,1,"./data/train-images.idx3-ubyte","./data/train-labels.idx1-ubyte",TRAINING_SIZE,1,1.0);

    
    free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
}