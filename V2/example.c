
#include <stdio.h>
#include <math.h>

#include "my_mnist.h"
#include "convolution.h"
#include "bayesianNetwork.h"
#include "convolutionalBayesianNetwork.h"
#include "trainNumberNodes.h"
#include "inference.h"
#include "em_algorithm.h"


#define TRAINING_SIZE 1000

int main(void){

    srand(2);

    
    printf("Loading data...\n");
    bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE,0.3);
    printf("finished reading images\n Reading Labels ... \n");
    int * labels = readLabels("./data/train-labels.idx1-ubyte", TRAINING_SIZE);
    printf("Done!\n");

    printf("MAIN: init cbn\n");


    ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2);
    cbn->bayesianNetworks[0] = createBayesianNetwork(28,1,1,true);
    addLayerToCbn(cbn,1,7,mustTMustFEither,3,2,4, true); 
    //addLayerToCbn(cbn,2,7,mustTMustFEither,3,2,7, true); 
    //addLayerToCbn(cbn,3,7,mustTMustFEither,3,2,10, true); 
    //addLayerToCbn(cbn,4,8,mustTMustFEither,2,2,13,false); 

    float thresholds[] = {0.2,0.3,0.4};
    int n_relations =  (int) ( log(TRAINING_SIZE / 100.0) / log(2.0));
    printf("n_relations = %d\n",n_relations);
    for (int i = 0; i < cbn->n_layers; i++){
        if (i == 0){
            addAllDependencies(cbn->bayesianNetworks[0],1,true);
        }else {
            optimizeKernelsAndStructure(cbn,i, images, TRAINING_SIZE,n_relations,true);
        }

        learnStructureNumberNodes(10,3,i,cbn,images,labels,TRAINING_SIZE,true);

        tuneCPTwithAugmentedData(cbn, i , "./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte" ,600 , 1, thresholds, 1,1.0);
    }

    printNumberNode(cbn->bayesianNetworks[0]->numberNodes[0],true);
    printNumberNode(cbn->bayesianNetworks[1]->numberNodes[0],true);
    printNumberNode(cbn->bayesianNetworks[1]->numberNodes[1],true);
    printNumberNode(cbn->bayesianNetworks[1]->numberNodes[2],true);

    free(labels);
    freeImages(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
    return 0;

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
    while (1){
        printf("iterations for gibbs sampling:\n");
        scanf("%d",&iterations);
        if (iterations == 0){
            break;
        }
        
        //setStateToImage(cbn,images[rand() % TRAINING_SIZE ]);
        setToRandomState(cbn,0.7);
        int n_samples = 99;
        bool *** samples = gibbsSampling(cbn,n_samples,iterations);
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


    freeImages(images,TRAINING_SIZE,28);
    freeConvolutionalBayesianNetwork(cbn);
}  
