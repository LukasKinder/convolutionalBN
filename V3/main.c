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
#include "inference.h"

#define TRAINING_SIZE 6000
#define N_NUMBER_NODES 0
#define N_INCOMING_RELATIONS 8

#define TRAIN_IMAGE_PATH "./data/train-images.idx3-ubyte"
#define TRAIN_Label_PATH "./data/train-labels.idx1-ubyte"

#define TEST_IMAGE_PATH "./data/t10k-images.idx3-ubyte"
#define TEST_Label_PATH "./data/t10k-labels.idx1-ubyte"

#define KERNEL_SEARCH_ITERATIONS 3000
#define COUNTS_UPDATE 300
#define STRUCTURE_UPDATE 1000
#define BATCH_SIZE 30

#define LEARNING_RATE 0.0002
#define MOMENTUM 0.5

int main(void){

    srand(33);

    float *** images = readImagesContinuos(TRAIN_IMAGE_PATH,TRAINING_SIZE);
    int * labels = readLabels(TRAIN_Label_PATH, TRAINING_SIZE);

    ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2);

    //layer 1
    addLayerToCbn(cbn,10,3,2, N_NUMBER_NODES ,4,true);
    addLayerToCbn(cbn,10,3,2, N_NUMBER_NODES ,7,true);

    //initKernels(cbn,0,images,20,false);

    //loadKernels(cbn->transitionalKernels[0], cbn->n_kernels[0],"hardcoded_layer1_n10_d1_s2");
    loadKernels(cbn->transitionalKernels[0], cbn->n_kernels[0],"layer1_n10_d1_s3");
    loadKernels(cbn->transitionalKernels[1], cbn->n_kernels[1],"layer2_n10_d10_s3");
    //loadKernels(cbn->transitionalKernels[2], cbn->n_kernels[2],"layer3_n20_d10_s2");
    
    /*loadKernels(cbn->transitionalKernels[2], cbn->n_kernels[2],"layer3_n10_d10_s3"); */
    

    /* saveKernelResponsesOfImage(cbn, images[0], "before");

    kernelTrainingWhileUpdatingStructure(cbn,2,KERNEL_SEARCH_ITERATIONS,COUNTS_UPDATE,STRUCTURE_UPDATE
            ,LEARNING_RATE,MOMENTUM,BATCH_SIZE,images,labels,TRAINING_SIZE,N_INCOMING_RELATIONS,TRAINING_SIZE,false); */

    //printKernel(cbn->transitionalKernels[0][0]);

    //saveLearningCurve(cbn->bayesianNetworks[2], "firstExperiment");

    //saveKernelResponsesOfImage(cbn, images[0], "after");
    /* saveKernelResponsesOfImage(cbn, images[1], "after1");
    saveKernelResponsesOfImage(cbn, images[2], "after2");
    saveKernelResponsesOfImage(cbn, images[3], "after3"); */

    //saveKernels(cbn->transitionalKernels[2],20,"layer3_n20_d10_s2");


    /* free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);

    return 0;  */

    printf("optimize structure layer 0\n");
    optimizeStructure(cbn,0,N_INCOMING_RELATIONS,images,labels,TRAINING_SIZE,10,0.005,50,true); 
    optimizeStructure(cbn,1,N_INCOMING_RELATIONS,images,labels,TRAINING_SIZE,10,0.005,50,true); 
    //optimizeStructure(cbn,2,N_INCOMING_RELATIONS,images,labels,TRAINING_SIZE,10,0.005,50,true); 


    //saveKernelResponsesOfImage(cbn, images[0]);

    printf("tune cbn with (augmented) data:\n");

    tuneCPTwithAugmentedData(cbn,0,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,TRAINING_SIZE,0,1.0,true);
    tuneCPTwithAugmentedData(cbn,1,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,TRAINING_SIZE,0,1.0,true);
    //tuneCPTwithAugmentedData(cbn,2,TRAIN_IMAGE_PATH,TRAIN_Label_PATH,TRAINING_SIZE,0,1.0,true);

    //printNode(cbn->bayesianNetworks[0]->nodes[0][7][21],true);

    //printf("Accuracy = %.3f\n",predictNumberAccuracy(cbn,TEST_IMAGE_PATH,TEST_Label_PATH,2000,true));

    /* printf("calculate rotating Accuracy\n");

    printf("Rotationg accuracy = %f\n",rotatingImageAccuracy(cbn,TEST_IMAGE_PATH, TEST_Label_PATH,10000,true));

    int s_means = cbn->bayesianNetworks[1]->size;
    float ** means = layerActivationImage(cbn->bayesianNetworks[1],0);
    char name[50] = "activations";
    saveImage(means,s_means,name,false);
    freeImageContinuos(means,s_means ); */

    /* printf("gibbs sampling\n");
    int n_samples = 1000;
    float *** gibbsSamples = strictClimbing(cbn, n_samples,images[0],100000);
    char name[50] = "gibbsSampleX";
    for (int i = 0; i < n_samples; i++){
        name[11] = '0' + i;
        saveImage(gibbsSamples[i],28,name,false);
    }
    freeImagesContinuos(gibbsSamples,n_samples,28); */

    printf("reconstruction Task\n");
    printf("%f\n", obstructionTask(cbn,images[0],15000,8 ));

    
    free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
    return 0;
}