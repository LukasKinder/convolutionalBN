

typedef struct RawConvolutionalBayesianNetwork *ConvolutionalBayesianNetwork;

typedef struct RawConvolutionalBayesianNetwork {
    BayesianNetwork* bayesianNetworks; //nodes within the x,y,depth grid
    int n_layers;
    Kernel ** transitionalKernels; //size n_layers-1; A collection of either MIMUE or weighted kernels
    Kernel * poolingKernels; //size n_layers; The final pooling kernels
    int * n_kernels; //n_kernels for each transitions
}RawConvolutionalBayesianNetwork;

ConvolutionalBayesianNetwork createConvolutionalBayesianNetwork(int n_layers, int* n_kernels, Kernel ** kernels, Kernel * poolingKernels){
    ConvolutionalBayesianNetwork cbn = malloc(sizeof(RawConvolutionalBayesianNetwork));
    cbn->n_layers = n_layers;
    cbn->bayesianNetworks = malloc(sizeof(BayesianNetwork) * n_layers);
    for (int i = 0; i < n_layers; i++){
        cbn->bayesianNetworks[i] = NULL;
    }
    cbn->n_kernels = n_kernels;
    cbn->transitionalKernels = kernels;
    cbn->poolingKernels = poolingKernels;
}

bool **** dataTransition(bool **** dataPrevious, int n_instances, int depth, int size, Kernel *kernels, int n_kernels, Kernel poolingKernel){
    bool **** newData = malloc(sizeof(bool***) * n_instances);
    bool **** temp = malloc(sizeof(bool***) * n_instances);;
    int sizeAfterNormalConvolution = sizeAfterConvolution(size, kernels[0]); //assume that all kernels lead to same size
    Kernel k;

    for (int  i = 0; i < n_instances; i++){
        newData[i] = malloc(sizeof(bool **) * n_kernels);
    }

    #pragma omp parallel for
    for (int  i = 0; i < n_instances; i++){
        for (int j = 0; j < n_kernels;j++){
            k = kernels[j]; 
            newData[i][j] = applyConvolution(dataPrevious[i],size,k);
        }
        temp[i] = newData[i];
        newData[i] = applyMaxPooling(temp[i],sizeAfterNormalConvolution,poolingKernel);
        freeImages(temp[i],n_kernels,sizeAfterNormalConvolution);
    }
    free(temp);
    return newData;
}

//learns the kernels and writes down counts for next layer
//if learn stucture is off, bayesian network in next layer will be fully connected
void learnLayer(ConvolutionalBayesianNetwork cbn,int layer, bool **** images, int n_images, int image_size,int neighbourDistance, bool learnStructure, bool learnKernels, bool diagonalRelations){
    int new_layer_size = image_size;
    int depth = (layer == 0 ? 1 : cbn->n_kernels[layer-1]);
    for (int i = 0; i < layer; i++){
        new_layer_size = sizeAfterConvolution(new_layer_size, cbn->transitionalKernels[i][0]);
        new_layer_size = sizeAfterConvolution(new_layer_size, cbn->poolingKernels[i]);
    }

    BayesianNetwork bn = createBayesianNetwork(new_layer_size,depth);
    if (learnStructure){
        printf("Structure learning not implemented!");
        exit(1);
        //init kernels & dependencies

        if (learnKernels){
            // Repeat;
            // optimize kernels given dependencies
            // optimize dependencies, given kernel
        }else{
            // optimize dependencies, given kernel
        }
    }else{


        addAllDependencies(bn,neighbourDistance,diagonalRelations);
        if (learnKernels){
            //optimize kernels given dependencies
        }
    }

    cbn->bayesianNetworks[layer] = bn;

    bool **** dataLayer = images;
    bool **** tmp;
    int temp_d = 1;
    int temp_size = image_size;
    for (int i = 0; i < layer; i++){
        tmp = dataLayer;
        dataLayer = dataTransition(tmp,n_images,temp_d,image_size,cbn->transitionalKernels[i],cbn->n_kernels[i],cbn->poolingKernels[i]);

        if (i != 0){
            //do not free images
            freeLayeredImages(tmp, n_images,temp_d,temp_size);
        }

        temp_d = cbn->n_kernels[i];
        temp_size = sizeAfterConvolution(temp_size,cbn->transitionalKernels[i][0]);
        temp_size = sizeAfterConvolution(temp_size,cbn->poolingKernels[i]);

    }


    fitDataCounts(bn,dataLayer,n_images);

    if (layer != 0){
        freeLayeredImages(dataLayer,n_images,temp_d,temp_size); //dont need to remember the training data anymore
    }
}

bool ** getImageFromState(ConvolutionalBayesianNetwork cbn){
    int image_size = cbn->bayesianNetworks[0]->size;
    bool ** image = malloc(sizeof(bool*) * image_size);
    for (int i = 0; i < image_size; i++){
        image[i] = malloc(sizeof(bool) * image_size);
        for (int j = 0; j < image_size; j++){
            image[i][j] = cbn->bayesianNetworks[0]->nodes[0][i][j]->value;
        }
    }
    return image;
}


void setStateToImage(ConvolutionalBayesianNetwork cbn ,bool *** layeredImage){
    BayesianNetwork bn;
    bool *** data = layeredImage;
    bool *** tmp;
    int currentSize = 28;
    Kernel k;

    setStateToData(cbn->bayesianNetworks[0], data);
    if (cbn->n_layers == 1){
        return;
    }

    for (int i = 0; i < cbn->n_layers-1; i++){

        bn = cbn->bayesianNetworks[i+1];

        tmp = data;

        data = malloc(sizeof(bool**) * bn->depth);

        for (int j = 0; j < cbn->n_kernels[i]; j++){
            k = cbn->transitionalKernels[i][j]; 
            data[j] = applyConvolution(tmp,currentSize,k);
        }
        currentSize = sizeAfterConvolution(currentSize,k); //does not matter which kernel exactly

        if (i != 0){
            freeImages(tmp, bn->depth,currentSize);
        }
        tmp = data;
        data = applyMaxPooling(tmp,currentSize,cbn->poolingKernels[i]);
        freeImages(tmp, bn->depth,currentSize);
        currentSize = bn->size;
        setStateToData(bn, data);
    }

    freeImages(data, bn->depth,currentSize);
}

void setToRandomState(ConvolutionalBayesianNetwork cbn, float fractionBlack){
    bool *** randomLayeredImage = malloc(sizeof(bool **) *1);
    randomLayeredImage[0] = malloc(sizeof(bool*) * 28);
    for (int i = 0; i < 28; i++){
        randomLayeredImage[0][i] = malloc(sizeof(bool) * 28);
        for (int j = 0; j < 28; j++){
            randomLayeredImage[0][i][j] = (float)(rand()) / (float)(RAND_MAX) > fractionBlack;
        }
    }
    setStateToImage(cbn,randomLayeredImage);
    freeImages(randomLayeredImage,1,28);
}


void freeConvolutionalBayesianNetwork(ConvolutionalBayesianNetwork cbn){

     for (int i = 0; i < cbn->n_layers -1; i++){
        for (int j = 0; j < cbn->n_kernels[i]; j++){
            freeKernel(cbn->transitionalKernels[i][j]);
        }
        free(cbn->transitionalKernels[i]);
    } 


    free(cbn->n_kernels);
    free(cbn->transitionalKernels);
    free(cbn->poolingKernels); 

    for (int i = 0; i < cbn->n_layers; i++){
        if (cbn->bayesianNetworks[i] != NULL) freeBayesianNetwork(cbn->bayesianNetworks[i]);   
    }
    free(cbn->bayesianNetworks);
    
    free(cbn);
}