
#include <stdio.h>

#include "my_mnist.h"
#include "convolution.h"
#include "bayesianNetwork.h"

#define TRAINING_SIZE 60000

int main(void){

    srand(42);

    if (true){

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

        addAllDependencies(bn);

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
    }
    return 0;
}  
