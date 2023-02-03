


typedef struct RawConvolutionalPerceptronModel *ConvolutionalPerceptronModel;

typedef struct RawConvolutionalPerceptronModel {
    PerceptronGrid* perceptronGrids; //nodes within the x,y,depth grid
    int n_implemented_layers;
    int n_layers;
    Kernel ** transitionalKernels; //size n_layers-1; A collection of either MIMUE or weighted kernels
    Kernel * poolingKernels; //size n_layers; The final pooling kernels
    int * n_kernels; //n_kernels for each transitions
}RawConvolutionalPerceptronModel;

void freeConvolutionalPerceptronModel(ConvolutionalPerceptronModel cpm){

     for (int i = 0; i < cpm->n_implemented_layers; i++){
        for (int j = 0; j < cpm->n_kernels[i]; j++){
            freeKernel(cpm->transitionalKernels[i][j]);
        }
        freeKernel(cpm->poolingKernels[i]);
        free(cpm->transitionalKernels[i]);
    } 


    free(cpm->n_kernels);
    free(cpm->transitionalKernels);
    free(cpm->poolingKernels); 
    

    for (int i = 0; i < cpm->n_implemented_layers; i++){
        if (cpm->perceptronGrids[i] != NULL) freePerceptronGrid(cpm->perceptronGrids[i]);   
    }
    free(cpm->perceptronGrids);
    
    free(cpm);
}

ConvolutionalPerceptronModel createConvolutionalPerceptronModel(int n_layers){
    ConvolutionalPerceptronModel cpm = malloc(sizeof(RawConvolutionalPerceptronModel ));
    cpm->n_layers = n_layers;
    cpm->perceptronGrids = malloc(sizeof(PerceptronGrid) * n_layers);
    
    cpm->n_kernels = malloc(sizeof(int) * n_layers);
    cpm->transitionalKernels = malloc(sizeof(Kernel * ) * n_layers);
    cpm->poolingKernels = malloc(sizeof(Kernel) * n_layers);
    cpm->n_implemented_layers = 0;
}

void initKernels(ConvolutionalPerceptronModel cpm, int layer, float ***test_images, int n_test_images, bool verbose){

    int d = 1,s = 28;
    float ****layered_images, **** temp;
    layered_images = imagesToLayeredImagesContinuos(test_images,n_test_images,28);
    for (int l = 0; l < layer; l++){
        temp = layered_images;
        layered_images = dataTransition(temp,n_test_images,d,s,cpm->transitionalKernels[l],cpm->n_kernels[l],cpm->poolingKernels[l]);

        freeLayeredImagesContinuos(temp,n_test_images,d,s);

        d = cpm->perceptronGrids[l]->depth;
        s = cpm->perceptronGrids[l]->size;
    }

    if (verbose) printf("INIT_KERNELS: data init done\n");

    Kernel k,k_old;
    for (int i = 0; i < cpm->n_kernels[layer]; i++){
        k_old = cpm->transitionalKernels[layer][i];

        if (verbose) printf("INIT_KERNELS: create promising kernel %d of %d\n",i,cpm->n_kernels[layer]);

        cpm->transitionalKernels[layer][i] = createPromisingKernel(k_old.size,k_old.depth,k_old.stride,k_old.padding, layered_images,s,n_test_images,verbose);

        freeKernel(k_old);
    }
    
    freeLayeredImagesContinuos(layered_images,n_test_images,d,s);
    if (verbose) printf("INIT_KERNELS: Done\n");
}


void loadKernels(Kernel *kernels, int n_kernels, char * name){
    char save_path[255] = "PretrainedKernels/";
    strncat(save_path,name, strlen(name) );

    int kernel_depth = kernels[0].depth;
    int kernel_size = kernels[0].size;

    FILE *fptr;
    fptr = fopen(save_path,"r");

    if(fptr == NULL){
        printf("Error! cant open %s\n",save_path);   
        exit(1);             
    }

    int file_n_kernels, filer_kernel_depth, file_size_kernels;

    fscanf(fptr,"%d %d %d",&file_n_kernels, &filer_kernel_depth, &file_size_kernels);


    if (file_n_kernels < n_kernels ||filer_kernel_depth != kernel_depth || file_size_kernels != kernel_size){

        printf("ERROR!! dimensions of kernel do not fit\n");
        printf("File: n %d depth %d size%d\n", file_n_kernels,filer_kernel_depth,file_size_kernels);
        printf("Template: n %d depth %d size%d\n", n_kernels,kernel_depth,kernel_size);
        exit(1);
    }

    if (file_n_kernels > n_kernels ){
        printf("WARNING: only a fraction of the loaded kernels are used\n");
    }

    for (int i = 0; i < n_kernels; i++){
        fscanf(fptr,"%f\n",&kernels[i].bias);
        for (int d = 0; d < kernel_depth; d++){
            for (int x = 0; x < kernel_size; x++){
                for (int y =0 ; y < kernel_size; y++){
                    fscanf(fptr, "%f ",&kernels[i].weights[d][x][y]);
                }
            }
        }
    }

    fclose(fptr);
}


void saveKernels(Kernel *kernels, int n_kernels, char * name){
    char save_path[255] = "PretrainedKernels/";
    strncat(save_path,name, strlen(name) );

    printf("save oath is: %s\n",name);

    int kernel_depth = kernels[0].depth;
    int kernel_size = kernels[0].size;

    FILE *fptr;
    fptr = fopen(save_path,"w");

    if(fptr == NULL){
        printf("Error when saving kernels! cant open %s\n",save_path);  
        exit(1);             
    }

    fprintf(fptr,"%d %d %d\n",n_kernels,kernel_depth,kernel_size);
    for (int i = 0; i < n_kernels; i++){
        fprintf(fptr,"%f\n",kernels[i].bias);
        for (int d = 0; d < kernel_depth; d++){
            for (int x = 0; x < kernel_size; x++){
                for (int y =0 ; y < kernel_size; y++){
                    fprintf(fptr, "%f ",kernels[i].weights[d][x][y]);
                }
                fprintf(fptr, "\n");
            }
            fprintf(fptr, "\n");
        }
        fprintf(fptr, "\n");
    }

    fclose(fptr);
}

//Todo enable stride and padding
void addLayerToCpm(ConvolutionalPerceptronModel cpm, int n_kernels, int kernel_size, int pooling_size, int max_distance_relation
        ,float gaussian_distance,float gaussian_width){
    Kernel * kernels = malloc(sizeof(Kernel) * n_kernels);
    int previous_data_depth = cpm->n_implemented_layers == 0 ? 1 : cpm->perceptronGrids[cpm->n_implemented_layers -1]->depth;
    for (int i = 0; i < n_kernels; i++){
        kernels[i] = createKernel(kernel_size,previous_data_depth, weighted,1,false);
    }

    cpm->transitionalKernels[cpm->n_implemented_layers] = kernels;
    cpm->n_kernels[cpm->n_implemented_layers] = n_kernels;
    cpm->poolingKernels[cpm->n_implemented_layers] = createKernel(pooling_size,n_kernels,pooling,1,false);

    int data_size = cpm->n_implemented_layers == 0 ? 28 :  cpm->perceptronGrids[cpm->n_implemented_layers -1]->size;
    data_size = sizeAfterConvolution(data_size,kernels[0]);
    data_size = sizeAfterConvolution(data_size,cpm->poolingKernels[cpm->n_implemented_layers]);

    cpm->perceptronGrids[cpm->n_implemented_layers] = createPerceptronGrid(data_size,n_kernels,max_distance_relation,gaussian_width,gaussian_distance); 
    cpm->n_implemented_layers++;
}


void setStateToImage(ConvolutionalPerceptronModel  cpm ,float ** image, bool fill_predicted_values){

    int d = 1, size = 28;

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

    for (int l = 0; l < cpm->n_layers; l++){
        after_transitional = malloc(sizeof(float **) * cpm->n_kernels[l]);
        for (int i = 0; i < cpm->n_kernels[l] ; i++){
            after_transitional[i] = applyConvolutionWeighted(before,size,size,cpm->transitionalKernels[l][i],true);
        }
        freeImagesContinuos(before,d,size);
        d = cpm->n_kernels[l];
        size = sizeAfterConvolution(size,cpm->transitionalKernels[l][0]);

        after_pooling = applyMaxPooling(after_transitional,size,cpm->poolingKernels[l]);
        freeImagesContinuos(after_transitional,d,size);
        size = sizeAfterConvolution(size,cpm->poolingKernels[l]);

        setStateToData(cpm->perceptronGrids[l],after_pooling,fill_predicted_values);

        before = after_pooling;
    }
    freeImagesContinuos(before,d,size);

}

void setToRandomState(ConvolutionalPerceptronModel cpm, bool fill_predicted_values){
    float ** randomImage = malloc(sizeof(float*) * 28);

    for(int i = 0; i < 28; i++){
        randomImage[i] = malloc(sizeof(float) * 28);
        for (int j = 0; j < 28; j++){
            randomImage[i][j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    
    setStateToImage(cpm,randomImage,fill_predicted_values);
    freeImageContinuos(randomImage,28);
}



void saveKernelResponsesOfImage(ConvolutionalPerceptronModel cpm, float ** image ,char * suffix){

    saveImage(image,28,"kernel_responses/ex_original_image",false);

    char name[255] = "kernel_responses/response_layerXX_KernelXX_";
    strncat(name,suffix, strlen(suffix));

    int d = 1, size = 28;
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
    for (int l = 0; l < cpm->n_layers; l++){
        after_transitional = malloc(sizeof(float **) * cpm->n_kernels[l]);
        for (int i = 0; i < cpm->n_kernels[l] ; i++){
            after_transitional[i] = applyConvolutionWeighted(before,size,size,cpm->transitionalKernels[l][i],true);
        }
        freeImagesContinuos(before,d,size);
        d = cpm->n_kernels[l];
        size = sizeAfterConvolution(size,cpm->transitionalKernels[l][0]);

        after_pooling = applyMaxPooling(after_transitional,size,cpm->poolingKernels[l]);
        freeImagesContinuos(after_transitional,d,size);
        size = sizeAfterConvolution(size,cpm->poolingKernels[l]);
        
        if (l == cpm->n_layers -1){
            for (int i = 0; i < d; i++){
                name[31] = (char)(l / 10 + '0');
                name[32] = (char)((l % 10)+ '0');

                name[40] = (char)(i / 10 + '0');
                name[41] = (char)((i % 10) + '0');

                printf("save %s\n",name);
                saveImage(after_pooling[i],size,name,true);
            }
        }

        before = after_pooling;
    }
    freeImagesContinuos(before,d,size);

}
