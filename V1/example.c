#include "my_mnist.h"
#include "convolution.h"

#define TRAINING_SIZE 2

int main(void){

    srand(17);

    if (false){
        bool *** images = readImages("./data/train-images.idx3-ubyte",TRAINING_SIZE);

        // print pixels of first data in test dataset
        printImage(images[0], 28);
        //Kernel k = createKernel(2,2,weighted,1,true);
        Kernel k = createKernel(2,2,mustTMustFEither,1,true);

        int sizeAfter = sizeAfterConvolution(28,k);
        printf("Size after convolution is %d \n", sizeAfter);


        bool** convolvedImage = applyConvolution(images,28, k);
        printKernel(k);
        printImage(convolvedImage, sizeAfter);

        freeKernel(k);
        freeImage(convolvedImage, sizeAfter );
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
