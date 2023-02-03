#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "my_mnist.h"
#include "convolution.h"
#include "perceptron_grid.h"
#include "convolutionalPerceptronModel.h"
#include "trainModel.h"

/*#include "performance_measure.h"
#include "inference.h" */



#define TRAIN_IMAGE_PATH "./data/train-images.idx3-ubyte"
#define TRAIN_Label_PATH "./data/train-labels.idx1-ubyte"
#define TEST_IMAGE_PATH "./data/t10k-images.idx3-ubyte"
#define TEST_Label_PATH "./data/t10k-labels.idx1-ubyte"

#define N_THREADS 16
#define TRAINING_SIZE 1000

int main(void){

    srand(5);

    float *** images = readImagesContinuos(TRAIN_IMAGE_PATH,TRAINING_SIZE);
    int * labels = readLabels(TRAIN_Label_PATH, TRAINING_SIZE);

    ConvolutionalPerceptronModel cpm = createConvolutionalPerceptronModel(2);

    //layer 1
    addLayerToCpm(cpm,10,2,2,7,4,1);
    addLayerToCpm(cpm,10,3,2,12,7,1);
    

    loadKernels(cpm->transitionalKernels[0], cpm->n_kernels[0],"hardcoded_layer1_n10_d1_s2");
    initKernels(cpm,1,images,20,false);

    setStateToImage(cpm,images[0],true);

    PerceptronGrid pg0 = cpm->perceptronGrids[0];
    printNode(pg0->nodes[4][14][12],true,pg0->depth,pg0->maxDistanceRelations);

    PerceptronGrid pg1= cpm->perceptronGrids[01];
    printNode(pg1->nodes[4][14][12],true,pg1->depth,pg1->maxDistanceRelations);


    free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalPerceptronModel(cpm);

    return 0; 
    
}