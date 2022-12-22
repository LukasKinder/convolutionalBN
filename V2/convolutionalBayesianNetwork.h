

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
    cbn->bayesianNetworks[0] = createBayesianNetwork(28,1,1,true); //use diagonal relation in pixel layer
    addAllDependencies(cbn->bayesianNetworks[0],1,true); //hardcoded relations in first layer
    cbn->n_kernels = NULL;
    cbn->transitionalKernels = NULL;
    cbn->poolingKernels = NULL;
}

//Todo enable stride and padding
void addLayerToCbn(ConvolutionalBayesianNetwork cbn, int n_kernels, KernelType kernel_type, int kernel_size
        , int pooling_size,int distance_relation, bool diagonals){
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
    cbn->bayesianNetworks[current_n_layers] = createBayesianNetwork(data_size, n_kernels,distance_relation,diagonals);

    (cbn->n_layers)++;
}

float logProbabilityStateCBN(ConvolutionalBayesianNetwork cbn){
    float prob = 0;
    for ( int i = 0; i < cbn->n_layers; i++){
        prob += logProbabilityStateBN(cbn->bayesianNetworks[i]);
    }
    return prob;
}

void tuneCPTwithAugmentedData(ConvolutionalBayesianNetwork cbn, int layer , char * path , int n_data, int shifts, float * thresholds, int n_thresholds, float pseudocounts){

    for(int i = 0; i < n_thresholds; i++){
        bool *** images = readImages("./data/train-images.idx3-ubyte",n_data,thresholds[i]);
        for(int shift_right = -shifts; shift_right < shifts +1;shift_right++ ){
            for(int shift_up = -shifts; shift_up < shifts +1;shift_up++ ){

                printf("add counts with th = %f, shiftRight = %d and shiftUp = %d\n",thresholds[i],shift_right,shift_up);

                bool ****layered_images, **** temp;
                int d,s;
                bool  *** shifted_images = shiftImages(images,n_data,28,shift_right,shift_up);
                layered_images = imagesToLayeredImages(shifted_images,n_data,28);
                freeImages(shifted_images,n_data,28);
                for (int l = 0; l < layer; l++){
                    temp = layered_images;
                    d = cbn->bayesianNetworks[l]->depth;
                    s = cbn->bayesianNetworks[l]->size;
                    layered_images = dataTransition(temp,n_data,d,s
                        ,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

                    freeLayeredImages(temp,n_data,d,s);
                }

                addDataCounts(cbn->bayesianNetworks[layer],layered_images,n_data);

                d = cbn->bayesianNetworks[layer]->depth;
                s = cbn->bayesianNetworks[layer]->size;
                freeLayeredImages(layered_images,n_data,d,s);


            }
        }
        freeImages(images,n_data,28);
    }
    fitCPTs(cbn->bayesianNetworks[layer],pseudocounts);
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
            data[j] = applyConvolution(tmp,currentSize,currentSize,k);
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

void printImageForEachKernel(bool ***layeredImage, int layer, int depth, int size){
    char name[25] = "rmageLayerXXKernelXX";
    for (int i  =0; i < depth; i++){
        name[10] = '0' + layer/10;
        name[11] = '0' + layer %10;
        name[18] = '0' + i / 10;
        name[19] = '0' + i %10;
        saveImage(layeredImage[i],size,name);
    }
}