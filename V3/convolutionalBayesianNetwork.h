

//predefine function
//void em_algorithm(BayesianNetwork bn, Kernel * kernels, int n_kernels, Kernel poolingKernel, bool **** data, int n_data, int size_data);

typedef struct RawConvolutionalBayesianNetwork *ConvolutionalBayesianNetwork;

typedef struct RawConvolutionalBayesianNetwork {
    BayesianNetwork* bayesianNetworks; //nodes within the x,y,depth grid
    int n_implemented_layers;
    int n_layers;
    Kernel ** transitionalKernels; //size n_layers-1; A collection of either MIMUE or weighted kernels
    Kernel * poolingKernels; //size n_layers; The final pooling kernels
    int * n_kernels; //n_kernels for each transitions
}RawConvolutionalBayesianNetwork;

void freeConvolutionalBayesianNetwork(ConvolutionalBayesianNetwork cbn){

     for (int i = 0; i < cbn->n_implemented_layers; i++){
        for (int j = 0; j < cbn->n_kernels[i]; j++){
            freeKernel(cbn->transitionalKernels[i][j]);
        }
        freeKernel(cbn->poolingKernels[i]);
        free(cbn->transitionalKernels[i]);
    } 


    free(cbn->n_kernels);
    free(cbn->transitionalKernels);
    free(cbn->poolingKernels); 
    

    for (int i = 0; i < cbn->n_implemented_layers; i++){
        if (cbn->bayesianNetworks[i] != NULL) freeBayesianNetwork(cbn->bayesianNetworks[i]);   
    }
    free(cbn->bayesianNetworks);
    
    free(cbn);
}

ConvolutionalBayesianNetwork createConvolutionalBayesianNetwork(int n_layers){
    ConvolutionalBayesianNetwork cbn = malloc(sizeof(RawConvolutionalBayesianNetwork));
    cbn->n_layers = n_layers;
    cbn->bayesianNetworks = malloc(sizeof(BayesianNetwork) * n_layers);
    
    cbn->n_kernels = malloc(sizeof(int) * n_layers);
    cbn->transitionalKernels = malloc(sizeof(Kernel * ) * n_layers);
    cbn->poolingKernels = malloc(sizeof(Kernel) * n_layers);
    cbn->n_implemented_layers = 0;
    
}

//Todo enable stride and padding
void addLayerToCbn(ConvolutionalBayesianNetwork cbn, int n_kernels, int kernel_size, int pooling_size, int n_number_nodes, int distance_relation, bool diagonals){
    Kernel * kernels = malloc(sizeof(Kernel) * n_kernels);
    int previous_data_depth = cbn->n_implemented_layers == 0 ? 1 : cbn->bayesianNetworks[ cbn->n_implemented_layers -1]->depth;
    for (int i = 0; i < n_kernels; i++){
        kernels[i] = createKernel(kernel_size,previous_data_depth, weighted,1,false);
    }

    

    cbn->transitionalKernels[cbn->n_implemented_layers] = kernels;
    cbn->n_kernels[cbn->n_implemented_layers] = n_kernels;
    cbn->poolingKernels[cbn->n_implemented_layers] = createKernel(pooling_size,n_kernels,pooling,1,false);

    int data_size = cbn->n_implemented_layers == 0 ? 28 :  cbn->bayesianNetworks[cbn->n_implemented_layers -1]->size;
    data_size = sizeAfterConvolution(data_size,kernels[0]);
    data_size = sizeAfterConvolution(data_size,cbn->poolingKernels[cbn->n_implemented_layers]);

    cbn->bayesianNetworks[cbn->n_implemented_layers] = createBayesianNetwork(data_size, n_kernels,n_number_nodes,distance_relation,diagonals);
    


    cbn->n_implemented_layers++;
}


void tuneCPTwithAugmentedData(ConvolutionalBayesianNetwork cbn, int layer , char * pathImages, char * pathLabels , int n_data, int shifts, float pseudocounts){

    BayesianNetwork bn;
    int n_parent_combinations;
    NumberNode nn;
    Node n;
    bn = cbn->bayesianNetworks[layer];

    //remove all counts from bayesian network
    for (int j = 0; j < bn->n_numberNodes; j++){
        nn = bn->numberNodes[j];
        n_parent_combinations = (int)(pow(2,nn->n_parents));
        if (nn->stateCounts == NULL){
            nn->stateCounts = malloc(sizeof(int *) * n_parent_combinations);
            for (int k = 0; k < n_parent_combinations; k++){
                nn->stateCounts[k] = malloc(sizeof(int) * 10);
            }
        }

        for (int k = 0; k < n_parent_combinations ; k++){
            for (int l  =0; l < 10; l++){
                nn->stateCounts[k][l] = 0;
            }
        }
    }
    for (int d = 0; d < bn->depth; d++){
        for(int x  =0; x < bn->size; x++){
            for (int y  =0; y < bn->size; y++){
                n = bn->nodes[d][x][y];
                n_parent_combinations = (int)(pow(2,n->n_parents));
                if (n->stateCountsTrue == NULL) n->stateCountsTrue = malloc(sizeof(int) * n_parent_combinations);
                if (n->stateCountsFalse == NULL) n->stateCountsFalse = malloc(sizeof(int) * n_parent_combinations);
                for (int j = 0; j < n_parent_combinations; j++){
                    n->stateCountsTrue[j]  =0;
                    n->stateCountsFalse[j]  =0; 
                }
            }
        }
    }
    
    int * labels = readLabels(pathLabels, n_data);
    float *** images = readImagesContinuos(pathImages,n_data);
    for(int shift_right = -shifts; shift_right < shifts +1;shift_right++ ){
        for(int shift_up = -shifts; shift_up < shifts +1;shift_up++ ){

            printf("add counts with shiftRight = %d and shiftUp = %d\n",shift_right,shift_up);

            float ****layered_images, **** temp;
            int d,s;
            float  *** shifted_images = shiftImagesContinuos(images,n_data,28,shift_right,shift_up);
            layered_images = imagesToLayeredImagesContinuos(shifted_images,n_data,28);
            freeImagesContinuos(shifted_images,n_data,28);
            for (int l = 0; l < layer+1; l++){
                temp = layered_images;
                if (l == 0){
                    d = 1;
                    s = 28;
                }else{
                    d = cbn->bayesianNetworks[l-1]->depth;
                    s = cbn->bayesianNetworks[l-1]->size;
                }

                printf("data transition: previous d: %d s: %d, kernel: %d pooling %d, n_kernels %d\n", d,s, cbn->transitionalKernels[l][0].size, cbn->poolingKernels[l].size,  cbn->n_kernels[l]);
                layered_images = dataTransition(temp,n_data,d,s
                    ,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

                freeLayeredImagesContinuos(temp,n_data,d,s);
            }

            addDataCounts(cbn->bayesianNetworks[layer],layered_images,n_data);
            #pragma omp parallel for
            for (int i_nn = 0; i_nn <  cbn->bayesianNetworks[layer]->n_numberNodes; i_nn++){
                addDataCountsNumberNode(cbn->bayesianNetworks[layer]->numberNodes[i_nn],layered_images,labels,n_data);
            }
            
            d = cbn->bayesianNetworks[layer]->depth;
            s = cbn->bayesianNetworks[layer]->size;
            
            freeLayeredImagesContinuos(layered_images,n_data,d,s);
        }
    }
    freeImagesContinuos(images,n_data,28);
    fitCPTs(cbn->bayesianNetworks[layer],pseudocounts);
    free(labels);
}


//ToDo rework? or remove?
/* bool ** getImageFromState(ConvolutionalBayesianNetwork cbn){
    int image_size = cbn->bayesianNetworks[0]->size;
    bool ** image = malloc(sizeof(bool*) * image_size);
    for (int i = 0; i < image_size; i++){
        image[i] = malloc(sizeof(bool) * image_size);
        for (int j = 0; j < image_size; j++){
            image[i][j] = cbn->bayesianNetworks[0]->nodes[0][i][j]->value;
        }
    }
    return image;
} */


void setStateToImage(ConvolutionalBayesianNetwork cbn ,float ** image){
    BayesianNetwork bn;
    float *** data = malloc(sizeof(bool **) *1);
    data[0] = image;
    float *** tmp;
    int currentSize = 28;
    Kernel k;

    setStateToData(cbn->bayesianNetworks[0], data);
    if (cbn->n_layers == 1){
        free(data);
        return;
    }

    for (int i = 1; i < cbn->n_layers; i++){

        bn = cbn->bayesianNetworks[i];
        tmp = data;
        data = malloc(sizeof(float**) * bn->depth);

        for (int j = 0; j < bn->depth; j++){
            k = cbn->transitionalKernels[i][j]; 
            data[j] = applyConvolutionWeighted(tmp,currentSize,currentSize,k);
        }

        if (i != 1){
            freeImagesContinuos(tmp, cbn->bayesianNetworks[i]->depth,cbn->bayesianNetworks[i]->size);
        }else{
            free(tmp);
        }
        
        currentSize = sizeAfterConvolution(currentSize,k); //does not matter which kernel exactly

        tmp = data;
        data = applyMaxPooling(tmp,currentSize,cbn->poolingKernels[i]);
        freeImagesContinuos(tmp, bn->depth,currentSize);
        currentSize = bn->size;
        setStateToData(bn, data);
    }

    freeImagesContinuos(data, bn->depth,currentSize);
}

void setToRandomState(ConvolutionalBayesianNetwork cbn){
    float ** randomImage = malloc(sizeof(float*) * 28);

    for(int i = 0; i < 28; i++){
        randomImage[i] = malloc(sizeof(float) * 28);
        for (int j = 0; j < 28; j++){
            randomImage[i][j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    
    setStateToImage(cbn,randomImage);
    freeImageContinuos(randomImage,28);
}



void saveKernelResponsesOfImage(ConvolutionalBayesianNetwork cbn, float ** image){
    char name[30] = "original_image";
    int d = 1, size = 28;
    saveImage(image,size,name,false);

    float *** before = malloc(sizeof(float **) * 1);
    before[0] = malloc(sizeof(float*) * size);
    for (int i  =0; i < 28; i++){
        before[0][i] = malloc(sizeof(float) * size);
        for (int j  = 0; j < size; j++){
            before[0][i][j] = image[i][j];
        }
    }

    
    float *** after_transitional;
    float *** after_pooling;
                        // 012345678901234567890
    char name_after[30] = "ex_layerXX_kernelXX_b";
    for (int l = 0; l < cbn->n_layers; l++){
        after_transitional = malloc(sizeof(float **) * cbn->n_kernels[l]);
        for (int i = 0; i < cbn->n_kernels[l] ; i++){
            after_transitional[i] = applyConvolutionWeighted(before,size,size,cbn->transitionalKernels[l][i]);
        }
        freeImagesContinuos(before,d,size);
        d = cbn->n_kernels[l];
        size = sizeAfterConvolution(size,cbn->transitionalKernels[l][0]);

        after_pooling = applyMaxPooling(after_transitional,size,cbn->poolingKernels[l]);
        freeImagesContinuos(after_transitional,d,size);
        size = sizeAfterConvolution(size,cbn->poolingKernels[l]);

        for (int i = 0; i < d; i++){
            name_after[8] = (char)(l / 10 + '0');
            name_after[9] = (char)(l + '0');

            name_after[17] = (char)(i / 10 + '0');
            name_after[18] = (char)(i + '0');

            name_after[20] = 'c';
            printf("save %s\n",name_after);
            saveImage(after_pooling[i],size,name_after,false);
            name_after[20] = 'b';
            printf("save %s\n",name_after);
            saveImage(after_pooling[i],size,name_after,true);
        }

        before = after_pooling;
    }
    freeImagesContinuos(before,d,size);

}