
#include <stdio.h>
#include <math.h>

#include "my_mnist.h"
#include "convolution.h"
#include "bayesianNetwork.h"
#include "convolutionalBayesianNetwork.h"
#include "trainNumberNodes.h"
#include "inference.h"
#include "em_algorithm.h"
#include "performance_measure.h"

#include "samplingVideo.h"


#define TRAINING_SIZE 6000

int main(void){

    srand(2);

    
    printf("Loading data...\n");
    bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE,0.5);
    printf("finished reading images\n Reading Labels ... \n");
    int * labels = readLabels("./data/train-labels.idx1-ubyte", TRAINING_SIZE);
    printf("Done!\n");

    printf("MAIN: init cbn\n");


    ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(3);
    cbn->bayesianNetworks[0] = createBayesianNetwork(28,1,1,true);
    addLayerToCbn(cbn,1,5,mustTMustFEither,3,2,4, true); 
    addLayerToCbn(cbn,2,10,mustTMustFEither,2,2,6, true); 
    //addLayerToCbn(cbn,3,7,mustTMustFEither,3,2,10, true); 
    //addLayerToCbn(cbn,4,8,mustTMustFEither,2,2,13,false); 

    float thresholds[] = {0.5,0.4,0.6};
    int n_relations =  (int) ( log(TRAINING_SIZE / 100.0) / log(2.0));
    printf("n_relations = %d\n",n_relations);
    for (int i = 0; i < cbn->n_layers; i++){
        if (i == 0){
            addAllDependencies(cbn->bayesianNetworks[0],1,true);
        }else {
            optimizeKernelsAndStructure(cbn,i, images, TRAINING_SIZE,4,true);
        }

        printf("Learn number nodes\n");

        learnStructureNumberNodes(1000,5,i,cbn,images,labels,TRAINING_SIZE,false);

        tuneCPTwithAugmentedData(cbn, i , "./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte" ,60000 , 0, thresholds, 1,1.0);
    }

    int s[4] = {2,0,2,3};

    generateImages(cbn,s,4,40000,100,2);

    free(labels);
    freeImages(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
    return 0;

    printf("calculating accuracy\n");

    printf("Accuracy = %f\n", predictNumberAccuracy(cbn, "./data/t10k-images.idx3-ubyte","./data/t10k-labels.idx1-ubyte", 10000 , true));


    predictNumberManualExperiment(cbn, images, TRAINING_SIZE, labels);


    printNumberNode(cbn->bayesianNetworks[0]->numberNodes[0],true);
    printNumberNode(cbn->bayesianNetworks[1]->numberNodes[1],true);
    printNumberNode(cbn->bayesianNetworks[1]->numberNodes[2],true);
    printNumberNode(cbn->bayesianNetworks[2]->numberNodes[3],true);
    printNumberNode(cbn->bayesianNetworks[2]->numberNodes[4],true);


    setStateToImage(cbn,images[TRAINING_SIZE -1]);
    int X,Y;
    for (int i = 0; i < 100; i++){
        printImage(images[TRAINING_SIZE -1],28);
        printf("give pixel cordinates\n");
        scanf("%d",&X);
        if (X == -1) break;
        scanf("%d",&Y);
        printf("%d %d: ",X,Y);
        probabilityPixelTrue(cbn,cbn->bayesianNetworks[0]->nodes[0][X][Y],true);
    }



    //saveBestWorst(images,TRAINING_SIZE,10,cbn);

    printf("MAIN: starting sampling\n");

    int iterations;
    int set_value_nn;
    while (1){
        printf("iterations for gibbs sampling:\n");
        scanf("%d",&iterations);
        if (iterations == 0){
            break;
        }

        printf("freeze value for number nodes:\n");
        scanf("%d",&set_value_nn);
        setValuesNumberNode(cbn,set_value_nn);
        
        //setStateToImage(cbn,images[rand() % TRAINING_SIZE ]);
        setToRandomState(cbn,0.7);
        int n_samples = 99;
        bool *** samples = gibbsSampling(cbn,n_samples,iterations, 1.0);
        char name[10] = "sampleXX";
        for (int i = 0; i < n_samples; i++){
            name[6] = (char)('0' +  i / 10 );
            name[7] = (char)('0' +  i % 10);
            saveImage(samples[i],28,name);
        }
        freeImages(samples,n_samples,28);
    }

    while (1){
        printf("iterations for SA sampling:\n");
        scanf("%d",&iterations);
        if (iterations == 0){
            break;
        }

        printf("freeze value for number nodes:\n");
        scanf("%d",&set_value_nn);
        setValuesNumberNode(cbn,set_value_nn);
        
        setStateToImage(cbn,images[rand() % TRAINING_SIZE]);
        //setToRandomState(cbn,0.7);
        int n_samples = 99;
        bool *** samples = simulatedAnnealing(cbn,n_samples,iterations);
        char name[10] = "sampleXX";
        for (int i = 0; i < n_samples; i++){
            name[6] = (char)('0' +  i / 10 );
            name[7] = (char)('0' +  i % 10);
            saveImage(samples[i],28,name);
        }
        freeImages(samples,n_samples,28);
    }

    while (1){
        printf("iterations for strict climbing:\n");
        scanf("%d",&iterations);
        if (iterations == 0){
            break;
        }

        printf("freeze value for number nodes:\n");
        scanf("%d",&set_value_nn);
        setValuesNumberNode(cbn,set_value_nn);
        
        setStateToImage(cbn,images[rand() % TRAINING_SIZE]);
        //setToRandomState(cbn,0.7);
        int n_samples = 99;

        bool *** samples = strictClimbing(cbn,n_samples,iterations);
        char name[10] = "sampleXX";
        for (int i = 0; i < n_samples; i++){
            name[6] = (char)('0' +  i / 10 );
            name[7] = (char)('0' +  i % 10);
            saveImage(samples[i],28,name);
        }
        freeImages(samples,n_samples,28);
    } 

    free(labels);
    freeImages(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
}  
