

//predefine function
void em_algorithm(BayesianNetwork bn, Kernel * kernels, int n_kernels, Kernel poolingKernel, bool **** data, int n_data, int size_data);

typedef struct RawConvolutionalBayesianNetwork *ConvolutionalBayesianNetwork;

typedef struct RawConvolutionalBayesianNetwork {
    BayesianNetwork* bayesianNetworks; //nodes within the x,y,depth grid
    int n_layers;
    Kernel ** transitionalKernels; //size n_layers-1; A collection of either MIMUE or weighted kernels
    Kernel * poolingKernels; //size n_layers; The final pooling kernels
    int * n_kernels; //n_kernels for each transitions
}RawConvolutionalBayesianNetwork;

ConvolutionalBayesianNetwork createConvolutionalBayesianNetwork(){
    ConvolutionalBayesianNetwork cbn = malloc(sizeof(RawConvolutionalBayesianNetwork));
    cbn->n_layers = 1;
    cbn->bayesianNetworks = malloc(sizeof(BayesianNetwork) * 1);
    cbn->bayesianNetworks[0] = createBayesianNetwork(28,1,1);
    addAllDependencies(cbn->bayesianNetworks[0],1,true); //hardcoded relations in first layer
    cbn->n_kernels = NULL;
    cbn->transitionalKernels = NULL;
    cbn->poolingKernels = NULL;
}

//Todo enable stride and padding
void addLayerToCbn(ConvolutionalBayesianNetwork cbn, int n_kernels, KernelType kernel_type, int kernel_size, int pooling_size){
    int current_n_layers = cbn->n_layers;
    Kernel * kernels = malloc(sizeof(Kernel) * n_kernels);
    int previous_data_depth = cbn->bayesianNetworks[current_n_layers-1]->depth;
    for (int i = 0; i < n_kernels; i++){
        kernels[i] = createKernel(kernel_size,previous_data_depth, kernel_type,1,false);
    }

    cbn->transitionalKernels = realloc(cbn->transitionalKernels,sizeof(Kernel *) * current_n_layers);
    cbn->transitionalKernels[current_n_layers-1] = kernels;

    cbn->n_kernels = realloc(cbn->n_kernels, sizeof(int) * (current_n_layers));
    cbn->n_kernels[current_n_layers-1] = n_kernels;

    cbn->poolingKernels = realloc(cbn->poolingKernels,sizeof(Kernel) * (current_n_layers));
    cbn->poolingKernels[current_n_layers-1] = createKernel(pooling_size,n_kernels,pooling,1,false);


    int data_size = cbn->bayesianNetworks[current_n_layers-1]->size;
    data_size = sizeAfterConvolution(data_size,kernels[0]);
    data_size = sizeAfterConvolution(data_size,cbn->poolingKernels[current_n_layers-1]);

    cbn->bayesianNetworks = realloc(cbn->bayesianNetworks,sizeof(BayesianNetwork) * (current_n_layers +1));
    int new_distance_relation = cbn->bayesianNetworks[current_n_layers-1]->distanceRelation + kernel_size + pooling_size -2;
    cbn->bayesianNetworks[current_n_layers] = createBayesianNetwork(data_size, n_kernels,new_distance_relation);

    (cbn->n_layers)++;
}

void fitCBN(ConvolutionalBayesianNetwork cbn, bool *** images, int n_data,float pseudo_Counts, bool verbose){

    if (verbose){
        printf("FIT_CBN: init layered image representation\n");
    }

    bool **** layeredImages;
    layeredImages = malloc(sizeof(bool ***) * n_data);
    for (int i =0; i < n_data; i++){
        layeredImages[i] = malloc(sizeof(bool**)*1);
        layeredImages[i][0] = images[i]; 
    }

    if (verbose) printf("FIT_CBN: fit layer 0\n");
    

    fitDataCounts(cbn->bayesianNetworks[0],layeredImages,n_data);
    fitCPTs(cbn->bayesianNetworks[0],pseudo_Counts,false,NULL,-1);

    if (verbose) printf("FIT_CBN: Done\n");
    

    if (cbn->n_layers == 1){
        for (int i =0; i < n_data; i++){
            free(layeredImages[i]);
        }
        free(layeredImages);
        return;
    }


    bool ****data = layeredImages;
    bool ****temp;
    for (int layer = 1; layer < cbn->n_layers; layer++){

        if (verbose) printf("FIT_CBN: fit layer %d using em-algorithm\n",layer);
        

        em_algorithm(cbn->bayesianNetworks[layer],cbn->transitionalKernels[layer-1],cbn->n_kernels[layer-1]
                ,cbn->poolingKernels[layer-1],data,n_data,cbn->bayesianNetworks[layer-1]->size);


        temp = data;
        
        data = dataTransition(temp,n_data,cbn->bayesianNetworks[layer-1]->depth,cbn->bayesianNetworks[layer-1]->size
            ,cbn->transitionalKernels[layer-1],cbn->n_kernels[layer-1],cbn->poolingKernels[layer-1]);

        printImageForEachKernel(data[0],layer,cbn->bayesianNetworks[layer]->depth,cbn->bayesianNetworks[layer]->size);
        

        if (verbose)printf("FIT_CBN: Learn CPTs\n");

        fitDataCounts(cbn->bayesianNetworks[layer],data,n_data);
        fitCPTs(cbn->bayesianNetworks[layer],pseudo_Counts,false,NULL,-1);

        if (verbose) printf("FIT_CBN: Done\n");
        
        
        if (layer == 1){
            for (int i =0; i < n_data; i++){
                free(layeredImages[i]);
            }
            free(layeredImages);
        } else {
            //printf("free previous %d %d %d \n",n_data,cbn->bayesianNetworks[layer-1]->depth,cbn->bayesianNetworks[layer-1]->size);
            freeLayeredImages(temp,n_data,cbn->bayesianNetworks[layer-1]->depth,cbn->bayesianNetworks[layer-1]->size);
        }
    }
    //printf("free last %d %d %d\n", n_data, cbn->bayesianNetworks[cbn->n_layers-1]->depth,cbn->bayesianNetworks[cbn->n_layers-1]->size);
    freeLayeredImages(data,n_data, cbn->bayesianNetworks[cbn->n_layers-1]->depth,cbn->bayesianNetworks[cbn->n_layers-1]->size);
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


void setStateToImage(ConvolutionalBayesianNetwork cbn ,bool ** image){
    BayesianNetwork bn;
    bool *** data = malloc(sizeof(bool **) *1);
    data[0] = image;
    bool *** tmp;
    int currentSize = 28;
    Kernel k;

    setStateToData(cbn->bayesianNetworks[0], data);
    if (cbn->n_layers == 1){
        free(data);
        return;
    }

    for (int i = 0; i < cbn->n_layers-1; i++){

        bn = cbn->bayesianNetworks[i+1];
        tmp = data;
        data = malloc(sizeof(bool**) * bn->depth);

        for (int j = 0; j < bn->depth; j++){
            k = cbn->transitionalKernels[i][j]; 
            data[j] = applyConvolution(tmp,currentSize,k);
        }

        if (i != 0){
            freeImages(tmp, cbn->bayesianNetworks[i]->depth,cbn->bayesianNetworks[i]->size);
        }else{
            free(tmp);
        }
        
        currentSize = sizeAfterConvolution(currentSize,k); //does not matter which kernel exactly

        tmp = data;
        data = applyMaxPooling(tmp,currentSize,cbn->poolingKernels[i]);
        freeImages(tmp, bn->depth,currentSize);
        currentSize = bn->size;
        setStateToData(bn, data);
    }

    freeImages(data, bn->depth,currentSize);
}

void setToRandomState(ConvolutionalBayesianNetwork cbn, float fractionBlack){
    bool ** randomImage = malloc(sizeof(bool*) * 28);

    for(int i = 0; i < 28; i++){
        randomImage[i] = malloc(sizeof(int) * 28);
        for (int j = 0; j < 28; j++){
            randomImage[i][j] = (float)(rand()) / (float)(RAND_MAX) > fractionBlack;
        }
    }
    
    setStateToImage(cbn,randomImage);
    freeImage(randomImage,28);
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