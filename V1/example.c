
#include <stdio.h>

#include "my_mnist.h"
#include "convolution.h"
#include "bayesianNetwork.h"
#include "convolutionalBayesianNetwork.h"
#include "inference.h"
#include "em_algorithm.h"
#define TRAINING_SIZE 100

int main(void){

    srand(42);

    if (true){+
        printf("Loading data...\n");
        bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE);
        printf("Done!\n");

        printf("MAIN: init cbn\n");
        ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork();

        addLayerToCbn(cbn,4,mustTMustFEither,3,2);
        addLayerToCbn(cbn,4,mustTMustFEither,3,2);
        addLayerToCbn(cbn,4,mustTMustFEither,3,2);
        addLayerToCbn(cbn,4,mustTMustFEither,2,2);
        addLayerToCbn(cbn,4,mustTMustFEither,2,2);

        printf("MAIN: fit cbn\n");
        fitCBN(cbn,images,TRAINING_SIZE,1,true);
        

        printf("MAIN: starting sampling\n");
        setStateToImage(cbn,images[0]);
        //setToRandomState(cbn,0.7);
        int n_samples = 55;
        bool *** samples = gibbsSampling(cbn,n_samples,100);
        //bool *** samples = simulatedAnnealing(cbn,n_samples,5000);
        //bool *** samples = strictClimbing(cbn,n_samples,5000);

        char name[10] = "sampleX";
        for (int i = 0; i < n_samples; i++){
            name[6] = i + '0';
            saveImage(samples[i],28,name);
        }

        freeImages(samples,n_samples,28);
        freeImages(images,TRAINING_SIZE,28);
        freeConvolutionalBayesianNetwork(cbn);
    }

    //calculate bic
    /*if (false){
        printf("Loading data...");
        bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE);
        printf("Done!\n");
        bool **** layeredImages;
        layeredImages = malloc(sizeof(bool ***) * TRAINING_SIZE);
        for (int i =0; i < TRAINING_SIZE; i++){
            layeredImages[i] = malloc(sizeof(bool**)*1);
            layeredImages[i][0] = images[i]; 
        }
        free(images);

        Kernel k1 = createKernel(2,1,mustTMustFEither,1,false);
        Kernel k2 = createKernel(2,1,mustTMustFEither,1,false);
        Kernel k3 = createKernel(2,1,mustTMustFEither,1,false);

        k1.map[0][0][0] = must_true;  k1.map[0][0][1] = either; 
        k1.map[0][1][0] = must_false; k1.map[0][1][1] = either; 

        k2.map[0][0][0] = must_true;  k2.map[0][0][1] = must_false; 
        k2.map[0][1][0] = either; k2.map[0][1][1] = either; 

        k3.map[0][0][0] = must_false;  k3.map[0][0][1] = must_false; 
        k3.map[0][1][0] = must_false; k3.map[0][1][1] = must_false; 

        Kernel *kernel_array = malloc(sizeof(Kernel) * 3);
        kernel_array[0] = k1; kernel_array[1] = k2;  kernel_array[2] = k3; 

        Kernel ** kernel_arrays = malloc(sizeof(Kernel * ) * 1);
        kernel_arrays[0] = kernel_array;

        int *n_kernels = malloc(sizeof(int) * 1);
        n_kernels[0] = 3;

        Kernel poolingKernel = createKernel(2,3,pooling,1,false);
        Kernel * pooling_kernels = malloc(sizeof(Kernel) * 1);
        pooling_kernels[0] = poolingKernel;

        printf("learning cbn\n");
        ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2,n_kernels,kernel_arrays,pooling_kernels);
        printf("Leanr layer0;\n");
        learnLayer(cbn,0,layeredImages,TRAINING_SIZE,28,1,false,false,true);
        printf("leanr layer 1:\n");
        learnLayer(cbn,1,layeredImages,TRAINING_SIZE,28,3,false,false,false); //neighbour distance to avoide overlapping responsive filters

        fitCPTs(cbn->bayesianNetworks[0],1,false,NULL,0);
        fitCPTs(cbn->bayesianNetworks[1],1,false,NULL,0);
        printf("finished learning cbn\n");


        bool **** dataLayer1;
        dataLayer1 = dataTransition(layeredImages,TRAINING_SIZE,1,28,cbn->transitionalKernels[0],cbn->n_kernels[0],cbn->poolingKernels[0]);

        float bic1 = bic(cbn->bayesianNetworks[0],layeredImages,TRAINING_SIZE,true);
        float bic2 = bic(cbn->bayesianNetworks[1],dataLayer1,TRAINING_SIZE,true);
        printf("BIC layer1:  %f, BIC layer2 = %f\n",bic1,bic2);

        float bic2A,bic2B,bic2C;
        bic2A = bicOneLevel(cbn->bayesianNetworks[1],dataLayer1,TRAINING_SIZE,0,true );
        bic2B = bicOneLevel(cbn->bayesianNetworks[1],dataLayer1,TRAINING_SIZE,1,true );
        bic2C = bicOneLevel(cbn->bayesianNetworks[1],dataLayer1,TRAINING_SIZE,2,true );
        printf("%f + %f + %f = %f\n",bic2A,bic2B,bic2C, bic2A+bic2B+bic2C);


        freeConvolutionalBayesianNetwork(cbn);
        freeLayeredImages(layeredImages,TRAINING_SIZE,1,28);
        freeLayeredImages(dataLayer1,TRAINING_SIZE,3,26);
    }

    //sampling two layers
    if (false){
        printf("Loading data...");
        bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE);
        printf("Done!\n");
        bool **** layeredImages;
        layeredImages = malloc(sizeof(bool ***) * TRAINING_SIZE);
        for (int i =0; i < TRAINING_SIZE; i++){
            layeredImages[i] = malloc(sizeof(bool**)*1);
            layeredImages[i][0] = images[i]; 
        }
        free(images);

        Kernel k1 = createKernel(2,1,mustTMustFEither,1,false);
        Kernel k2 = createKernel(2,1,mustTMustFEither,1,false);
        Kernel k3 = createKernel(2,1,mustTMustFEither,1,false);

        k1.map[0][0][0] = must_true;  k1.map[0][0][1] = either; 
        k1.map[0][1][0] = must_false; k1.map[0][1][1] = either; 

        k2.map[0][0][0] = must_true;  k2.map[0][0][1] = must_false; 
        k2.map[0][1][0] = either; k2.map[0][1][1] = either; 

        k3.map[0][0][0] = must_false;  k3.map[0][0][1] = must_false; 
        k3.map[0][1][0] = must_false; k3.map[0][1][1] = must_false; 

        Kernel *kernel_array = malloc(sizeof(Kernel) * 3);
        kernel_array[0] = k1; kernel_array[1] = k2;  kernel_array[2] = k3; 

        Kernel ** kernel_arrays = malloc(sizeof(Kernel * ) * 1);
        kernel_arrays[0] = kernel_array;

        int *n_kernels = malloc(sizeof(int) * 1);
        n_kernels[0] = 3;

        Kernel poolingKernel = createKernel(2,3,pooling,1,false);
        Kernel * pooling_kernels = malloc(sizeof(Kernel) * 1);
        pooling_kernels[0] = poolingKernel;

        printf("learning cbn\n");
        ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2,n_kernels,kernel_arrays,pooling_kernels);
        learnLayer(cbn,0,layeredImages,TRAINING_SIZE,28,1,false,false,true);
        learnLayer(cbn,1,layeredImages,TRAINING_SIZE,28,3,false,false,false); //neighbour distance to avoide overlapping responsive filters

        fitCPTs(cbn->bayesianNetworks[0],1,false,NULL,0);
        fitCPTs(cbn->bayesianNetworks[1],1,false,NULL,0);
        printf("finished learning cbn\n");

        int n_gibbs_samples = 10;

        //setStateToImage(cbn,layeredImages[4]);
        setToRandomState(cbn,0.7);
        
        printf("starting sampling\n");
        bool *** samples = gibbsSampling(cbn,10,10000);
        //bool *** samples = simulatedAnnealing(cbn,10,10000);
        //bool *** samples = strictClimbing(cbn,10,50000);
        printf("end sampling\n");

        saveImage(samples[0],28,"sample0");
        saveImage(samples[1],28,"sample1");
        saveImage(samples[2],28,"sample2");
        saveImage(samples[3],28,"sample3");
        saveImage(samples[4],28,"sample4");
        saveImage(samples[5],28,"sample5");
        saveImage(samples[6],28,"sample6");
        saveImage(samples[7],28,"sample7");
        saveImage(samples[8],28,"sample8");
        saveImage(samples[9],28,"sample9");

        freeImages(samples,n_gibbs_samples,28);
        freeConvolutionalBayesianNetwork(cbn);
        freeLayeredImages(layeredImages,TRAINING_SIZE,1,28);

    }

    //sampling one layer
    if (false){

        printf("Loading data...");
        bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE);
        printf("Done!\n");
        bool **** layeredImages;
        layeredImages = malloc(sizeof(bool ***) * TRAINING_SIZE);
        for (int i =0; i < TRAINING_SIZE; i++){
            layeredImages[i] = malloc(sizeof(bool**)*1);
            layeredImages[i][0] = images[i]; 
        }
        free(images);

        ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(1,NULL,NULL,NULL);


        learnLayer(cbn,0,layeredImages,TRAINING_SIZE,28,1,false,false,true);
        fitCPTs(cbn->bayesianNetworks[0],1,false,NULL,0);

        int n_gibbs_samples = 10;

        //setStateToImage(cbn,layeredImages[2]);
        setToRandomState(cbn,0.7);
        
        printf("starting sampling\n");
        //bool *** samples = simulatedAnnealing(cbn,10,50000);
        bool *** samples = strictClimbing(cbn,10,50000);
        printf("end sampling\n");

        saveImage(samples[0],28,"sample0");
        saveImage(samples[1],28,"sample1");
        saveImage(samples[2],28,"sample2");
        saveImage(samples[3],28,"sample3");
        saveImage(samples[4],28,"sample4");
        saveImage(samples[5],28,"sample5");
        saveImage(samples[6],28,"sample6");
        saveImage(samples[7],28,"sample7");
        saveImage(samples[8],28,"sample8");
        saveImage(samples[9],28,"sample9");

        freeImages(samples,n_gibbs_samples,28);
        freeConvolutionalBayesianNetwork(cbn);
        freeLayeredImages(layeredImages,TRAINING_SIZE,1,28);

    }

    if (false){

        printf("Loading data...");
        bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE);
        printf("Done!\n");

        bool **** layeredImages;
        layeredImages = malloc(sizeof(bool ***) * TRAINING_SIZE);
        for (int i =0; i < TRAINING_SIZE; i++){
            layeredImages[i] = malloc(sizeof(bool**)*1);
            layeredImages[i][0] = images[i]; 
        }
        free(images);

        Kernel k1 = createKernel(2,1,mustTMustFEither,1,false);
        Kernel k2 = createKernel(2,1,mustTMustFEither,1,false);
        Kernel k3 = createKernel(2,1,mustTMustFEither,1,false);

        k1.map[0][0][0] = must_true;  k1.map[0][0][1] = either; 
        k1.map[0][1][0] = must_false; k1.map[0][1][1] = either; 

        k2.map[0][0][0] = must_true;  k2.map[0][0][1] = must_false; 
        k2.map[0][1][0] = either; k2.map[0][1][1] = either; 

        k3.map[0][0][0] = must_false;  k3.map[0][0][1] = must_false; 
        k3.map[0][1][0] = must_false; k3.map[0][1][1] = must_false; 

        Kernel *kernel_array = malloc(sizeof(Kernel) * 3);
        kernel_array[0] = k1; kernel_array[1] = k2;  kernel_array[2] = k3; 

        Kernel ** kernel_arrays = malloc(sizeof(Kernel * ) * 1);
        kernel_arrays[0] = kernel_array;

        int *n_kernels = malloc(sizeof(int) * 1);
        n_kernels[0] = 3;

        Kernel poolingKernel = createKernel(2,3,pooling,1,false);
        Kernel * pooling_kernels = malloc(sizeof(Kernel) * 1);
        pooling_kernels[0] = poolingKernel;

        ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2,n_kernels,kernel_arrays,pooling_kernels);
        learnLayer(cbn,0,layeredImages,TRAINING_SIZE,28,1,false,false,true);
        learnLayer(cbn,1,layeredImages,TRAINING_SIZE,28,3,false,false,false); //neighbour distance to avoide overlapping responsive filters

        fitCPTs(cbn->bayesianNetworks[0],1,false,NULL,0);
        fitCPTs(cbn->bayesianNetworks[1],1,false,NULL,0);

        setStateToImage(cbn,layeredImages[0]);

        int n_effected_nodes;
        Node * effected_nodes = effectedNodesOfPixel(cbn,5,5,1,&n_effected_nodes);
        printf("%d nodes effected\n",n_effected_nodes);
        free(effected_nodes);

        //bool ** state = getImageFromState(cbn);
        //printImage(state,28);
        //freeImage(state,28);

        //printNode(cbn->bayesianNetworks[1]->nodes[0][12][12],true);

        //printNode(cbn->bayesianNetworks[1]->nodes[2][3][14],true);

        freeConvolutionalBayesianNetwork(cbn);

        freeLayeredImages(layeredImages,TRAINING_SIZE,1,28);
    }

    if (false){

        printf("Loading data...");
        bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE);
        printf("Done!\n");

        bool **** layeredImages;
        layeredImages = malloc(sizeof(bool ***) * TRAINING_SIZE);
        for (int i =0; i < TRAINING_SIZE; i++){
            layeredImages[i] = malloc(sizeof(bool**)*1);
            layeredImages[i][0] = images[i]; 
        }
        free(images);

        Kernel k1 = createKernel(2,1,mustTMustFEither,1,false);
        Kernel k2 = createKernel(2,1,mustTMustFEither,1,false);
        Kernel k3 = createKernel(2,1,mustTMustFEither,1,false);
        Kernel *kernel_array = malloc(sizeof(Kernel) * 3);
        kernel_array[0] = k1; kernel_array[1] = k2;  kernel_array[2] = k3; 

        k1.map[0][0][0] = must_true;  k1.map[0][0][1] = either; 
        k1.map[0][1][0] = must_false; k1.map[0][1][1] = either; 

        k2.map[0][0][0] = must_true;  k2.map[0][0][1] = must_false; 
        k2.map[0][1][0] = either; k2.map[0][1][1] = either; 

        k3.map[0][0][0] = must_false;  k3.map[0][0][1] = must_false; 
        k3.map[0][1][0] = must_false; k3.map[0][1][1] = must_false; 

        Kernel poolingKernel = createKernel(2,3,pooling,2,false);

        int size_after = sizeAfterConvolution(28,k1);
        size_after = sizeAfterConvolution(size_after,poolingKernel);
        printf("Size after (in main) is %d\n", size_after);

        bool **** new_data = dataTransition(layeredImages,TRAINING_SIZE,1,28, kernel_array,3,poolingKernel);

        printKernel(k1);
        printImage(new_data[0][0],size_after);
        printKernel(k2);
        printImage(new_data[0][1],size_after);
        printKernel(k3);
        printImage(new_data[0][2],size_after);

        freeKernel(k1); freeKernel(k2);freeKernel(k3);
        free(kernel_array);

        freeLayeredImages(layeredImages,TRAINING_SIZE,1,28);
        freeLayeredImages(new_data,TRAINING_SIZE,3,size_after);
    }

    if (false){

        Kernel k1 = createKernel(2,1,weighted,1,false);
        Kernel k2 = createKernel(2,1,weighted,1,false);
        Kernel k3 = createKernel(2,1,weighted,1,false);

        Kernel *kernel_array = malloc(sizeof(Kernel) * 3);
        kernel_array[0] = k1; kernel_array[1] = k2;  kernel_array[2] = k3; 

        Kernel ** kernel_arrays = malloc(sizeof(Kernel * ) * 1);
        kernel_arrays[0] = kernel_array;

        int *n_kernels = malloc(sizeof(int) * 1);
        n_kernels[0] = 3;

        Kernel poolingKernel = createKernel(2,3,pooling,2,false);
        Kernel * pooling_kernels = malloc(sizeof(Kernel) * 1);
        pooling_kernels[0] = poolingKernel;

        ConvolutionalBayesianNetwork cbn = createConvolutionalBayesianNetwork(2,n_kernels,kernel_arrays,pooling_kernels);

        freeConvolutionalBayesianNetwork(cbn);

        //freeImages(images, TRAINING_SIZE,28);
    }

    if (false){

        printf("Loading data...");
        bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE);
        printf("Done!\n");

        Kernel k1 = createKernel(2,1,weighted,1,false);
        Kernel k2 = createKernel(2,1,weighted,1,false);
        Kernel k3 = createKernel(2,1,weighted,1,false);
        int sizeAfter = sizeAfterConvolution(28,k1);

        bool **** dataLayer2 = malloc(sizeof(bool***) * TRAINING_SIZE);

        #pragma omp parallel for
        for (int i = 0; i < TRAINING_SIZE; i++){
            dataLayer2[i] = malloc(sizeof(bool**) * 3);
            
            dataLayer2[i][0] = applyConvolution(images,28, k1);
            dataLayer2[i][1] = applyConvolution(images,28, k2);
            dataLayer2[i][2] = applyConvolution(images,28, k3);
        }

        BayesianNetwork bn = createBayesianNetwork(sizeAfter,3);

        addAllDependencies(bn, false,1);

        printf("About to fit counts\n");
        fitDataCounts(bn,dataLayer2,TRAINING_SIZE);
        printf("Done\n");


        freeKernel(k1); freeKernel(k2); freeKernel(k3);
        freeBayesianNetwork(bn);
        freeImages(images, TRAINING_SIZE,28);
        freeLayeredImages(dataLayer2, TRAINING_SIZE,3,sizeAfter );

        printf("Done and freed everything\n");

    }

    if (false){
        bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE);

        // print pixels of first data in test dataset
        printImage(images[0], 28);
        //Kernel k = createKernel(2,2,weighted,1,true);
        Kernel k = createKernel(2,2,pooling,1,false);

        int sizeAfter = sizeAfterConvolution(28,k);
        printf("Size after convolution is %d \n", sizeAfter);

        //bool** convolvedImage = applyConvolution(images,28, k);
        bool*** convolvedLayers = applyMaxPooling(images,28,k);

        printKernel(k);
        printImage(convolvedLayers[0], sizeAfter);
        printImage(convolvedLayers[1], sizeAfter);

        saveImage(convolvedLayers[0], sizeAfter, "test");

        freeKernel(k);
        freeImages(convolvedLayers, 2, sizeAfter );
        freeImages(images, TRAINING_SIZE,28);
    }

    if (false){
        printf("A\n");
        Kernel k1 = createKernel(3,2,pooling,1,false);
        printf("B\n");
        Kernel k2 = createKernel(4,2,mustTMustFEither,1,false);
        printf("C\n");
        Kernel k3 = createKernel(5,2,weighted,1,false);
        printf("D\n");

        printKernel(k1);
        printKernel(k2);
        printKernel(k3);

        freeKernel(k1);
        freeKernel(k2);
        freeKernel(k3);
    }

    if (false){
        for (int pl = 0; pl < 2; pl ++){
            bool pooling = pl ==1 ? true : false;
            for (int kernelSize = 1; kernelSize < 5; kernelSize ++){
                for (int stride = 1; stride < 4; stride +=1){
                    Kernel kernel = createKernel(kernelSize,1,pooling,stride,pooling);
                    for (int data_size = 3; data_size < 10; data_size +=1){
                        int result = sizeAfterConvolution(data_size, kernel );
                        printf("padding = %s, kernel size = %d, stride = %d, data size = %d ===> %d\n", pooling ? "true" : "false", kernelSize,stride,data_size,result);
                    }
                    freeKernel(kernel);
                }
            }
        } 
    }*/
    return 0;
}  
