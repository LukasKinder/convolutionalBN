#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "my_mnist.h"
#include "convolution.h"
#include "perceptron_grid.h"
#include "convolutionalPerceptronModel.h"
#include "trainStructure.h"
#include "pretrain_kernels.h"
#include "trainKernels.h"
#include "performance_measure.h"
#include "inference.h"

#define TRAINING_SIZE 1000
#define TEST_SIZE 1000
#define N_NUMBER_NODES 0
#define N_INCOMING_RELATIONS 3

#define TRAIN_IMAGE_PATH "./data/train-images.idx3-ubyte"
#define TRAIN_Label_PATH "./data/train-labels.idx1-ubyte"

#define TEST_IMAGE_PATH "./data/t10k-images.idx3-ubyte"
#define TEST_Label_PATH "./data/t10k-labels.idx1-ubyte"

#define KERNEL_SEARCH_N_COUNTS_UPDATE 500
#define KERNEL_SEARCH_ITERATIONS 100
#define COUNTS_UPDATE 5
#define STRUCTURE_UPDATE 100
#define BATCH_SIZE 100

#define LEARNING_RATE 0.03
#define MOMENTUM 0.5

/* TODO 
Gradient of each kernel gets normalized individually
take care of proportion white 
Investigate uniform shift, and why still outgoing relations*/

int main(void){

    srand(5);

    float *** images = readImagesContinuos(TRAIN_IMAGE_PATH,TRAINING_SIZE);
    int * labels = readLabels(TRAIN_Label_PATH, TRAINING_SIZE);

    ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2);


    /* addLayerToCbn(cbn,1,1,1, N_NUMBER_NODES ,4,true);
    initKernels(cbn,0,images,20,false);
    cbn->transitionalKernels[0][0].weights[0][0][0] = 1;
    cbn->transitionalKernels[0][0].bias = 0; */


    //layer 1
    addLayerToCbn(cbn,10,3,2, N_NUMBER_NODES ,2,true);
    addLayerToCbn(cbn,10,3,2, N_NUMBER_NODES ,16,true);

    //initKernels(cbn,0,images,20,false);

    loadKernels(cbn->transitionalKernels[0], cbn->n_kernels[0],"overlapping_l0_n10_s3");
    //loadKernels(cbn->transitionalKernels[1],cbn->n_kernels[1],"overlapping_l1_n10_s3");

    repeatedly_replace_worse_kernels(cbn,1,images,TRAINING_SIZE,N_INCOMING_RELATIONS,30, true);



    /* float *** test_images = readImagesContinuos(TEST_IMAGE_PATH,TEST_SIZE);
    int * test_labels = readLabels(TEST_Label_PATH, TEST_SIZE);
    writeKernelResponsesToFile(cbn->transitionalKernels,cbn->n_kernels,cbn->poolingKernels,cbn->n_layers,labels,images,TRAINING_SIZE, "10000_two_kernels_Train",false);
    writeKernelResponsesToFile(cbn->transitionalKernels,cbn->n_kernels,cbn->poolingKernels,cbn->n_layers,test_labels,test_images,TEST_SIZE, "10000_two_kernels_Test",false); */

    /* free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);

    return 0;  */ 
    
    /*loadKernels(cbn->transitionalKernels[2], cbn->n_kernels[2],"layer3_n10_d10_s3"); */
    

    /*saveKernelResponsesOfImage(cbn, images[0], "before");
    saveKernelResponsesOfImage(cbn, images[1], "before1");
    saveKernelResponsesOfImage(cbn, images[2], "before2");
    saveKernelResponsesOfImage(cbn, images[3], "before3");
    saveKernelResponsesOfImage(cbn, images[4], "before4"); */

    saveKernels(cbn->transitionalKernels[1],cbn->n_kernels[1],"overlapping_l1_n10_s3_before");

    kernelTrainingWhileUpdatingStructure(cbn,1,KERNEL_SEARCH_ITERATIONS,COUNTS_UPDATE,STRUCTURE_UPDATE
            ,LEARNING_RATE,MOMENTUM,BATCH_SIZE,images,labels,TRAINING_SIZE,N_INCOMING_RELATIONS,KERNEL_SEARCH_N_COUNTS_UPDATE,true);

    //printKernel(cbn->transitionalKernels[0][0]);

    saveLearningCurve(cbn->bayesianNetworks[1], "firstExperiment");

    saveKernelResponsesOfImage(cbn, images[0], "after");
/*     saveKernelResponsesOfImage(cbn, images[1], "after1");
    saveKernelResponsesOfImage(cbn, images[2], "after2");
    saveKernelResponsesOfImage(cbn, images[3], "after3");
    saveKernelResponsesOfImage(cbn, images[4], "after4"); */

    saveKernels(cbn->transitionalKernels[1],cbn->n_kernels[1],"overlapping_l1_n10_s3");

    free(labels);
    freeImagesContinuos(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
    return 0;


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