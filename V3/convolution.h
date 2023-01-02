

float sigmoid(float x){
    return 1 / (1 + exp(-x));
}

typedef enum KernelType {
    pooling, weighted
}KernelType ;

typedef struct Kernel {
    int size;
    KernelType type;
    int stride;
    bool padding;
    int depth;

    //only used for Weighted Kernels:
    float bias;
    float *** weights; 
  
}Kernel;

//initializes a kernel with random weights
Kernel createKernel(int size,int depth, KernelType type, int stride, bool padding){
    Kernel kernel;
    kernel.size = size;
    kernel.type = type;
    kernel.stride = stride;
    kernel.padding = padding;
    kernel.depth = depth;

    if (type ==  weighted){
        kernel.weights = malloc(sizeof(float**) * depth);
        for (int i = 0; i < depth; i++){
            kernel.weights[i] = malloc(sizeof(float*) * size);
            for (int j = 0; j < size; j ++){
                kernel.weights[i][j] = malloc(sizeof(float) * size);
                for (int k = 0; k < size; k ++){
                    // choose weights with uniform(-1,1)
                    kernel.weights[i][j][k] = (float)rand()/(float)(RAND_MAX/2) -1; 
                }
            }
        }
        // choose bias with uniform(-1,1)
        kernel.bias = (float)rand()/(float)(RAND_MAX/2) -1; 
    }
    return kernel;
}

void printKernel(Kernel kernel){
    if (kernel.type == pooling){
        printf("Pooling kernel with size %d\n",kernel.size);
    } else {
        printf("weighted kernel with size %d and bias %.2f:\n",kernel.size, kernel.bias);
        for (int k = 0; k < kernel.depth; k++){
            printf("Depth %d:\n", k);
            for (int i = 0; i < kernel.size; i++){
                for (int j = 0; j < kernel.size; j ++){
                    if (kernel.weights[k][i][j] < 0){
                        printf("%.1f ",kernel.weights[k][i][j]);
                    }else {
                        printf(" %.1f ",kernel.weights[k][i][j]);
                    }
                }
                printf("\n");
            }
        }
    }
}

void freeKernel(Kernel kernel){
    if (kernel.type == weighted) {
        for (int i = 0; i < kernel.depth; i++){
            for (int j = 0; j < kernel.size; j++ ){
                free(kernel.weights[i][j]);
            }
            free(kernel.weights[i]);
        }
        free(kernel.weights);
    }
}

int sizeAfterConvolution(int originalSize, Kernel kernel){
    if (kernel.padding){
        originalSize += 2 * (kernel.size-1);
    }

    return (originalSize - kernel.size + 1) / kernel.stride ;
}

//depth of kernel may not be 1
float ** applyMaxPoolingOneLayer(float ** data, int data_size_x,int data_size_y, Kernel kernel){
    if (kernel.type != pooling){
        printf("wrong type of kernel for pooling");
    }

    int newSize_x = sizeAfterConvolution(data_size_x,kernel);
    int newSize_y = sizeAfterConvolution(data_size_y,kernel);

    float ** newData = malloc(sizeof(float *) * newSize_x);
    for (int i = 0; i < newSize_x; i++){
        newData[i] = malloc(sizeof(float) * newSize_y);
    }

    int i,j,responseX,responseY;
    float max_value;

    for (int x = 0; x < newSize_x; x++){
        for (int y = 0; y < newSize_y; y++){
            responseX = x * kernel.stride - (kernel.padding ? kernel.size -1: 0);
            responseY = y * kernel.stride - (kernel.padding ? kernel.size -1: 0);
            max_value = -999999999;
            for (i = responseX; i < responseX +  kernel.size; i++){
                for (j = responseY; j < responseY + kernel.size; j++){

                    if (i < 0 || j < 0 || i>=data_size_x || j >= data_size_y){
                        continue;; //out of bound because of padding
                    } else {
                        max_value = max_value < data[i][j] ? data[i][j] : max_value;
                    }
                }
            }
            newData[x][y] = max_value;
        }
    }
    
    return newData;

}


float *** applyMaxPooling(float ***data, int data_size, Kernel kernel){
    if (kernel.type != pooling){
        printf("wrong type of kernel for pooling");
    }

    float *** newData = malloc(sizeof(float **) * kernel.depth);

    #pragma omp parallel for
    for (int d = 0; d < kernel.depth; d++){
        newData[d] = applyMaxPoolingOneLayer(data[d],data_size,data_size,kernel);
    }

    return newData;
}

float ** applyConvolutionWeighted(float*** data, int data_size_x, int data_size_y, Kernel kernel){

    int responseX, responseY,x,y,i,j,d;
    int newSize_x = sizeAfterConvolution(data_size_x,kernel);
    int newSize_y = sizeAfterConvolution(data_size_y,kernel);
    float sum,value;

    float** new_data = malloc(sizeof(float*) * newSize_x);
    for (int i = 0; i < newSize_x; i++){
        new_data[i] = malloc(sizeof(float) * newSize_y);
    }

    for (x = 0; x < newSize_x; x++){
        for (y = 0; y < newSize_y; y++){

            responseX = x * kernel.stride - (kernel.padding ? kernel.size -1: 0);
            responseY = y * kernel.stride - (kernel.padding ? kernel.size -1: 0);

            sum = 0;
            for (d = 0; d < kernel.depth; d++){
                for (i = responseX; i < responseX +  kernel.size; i++){
                    for (j = responseY; j < responseY + kernel.size; j++ ){
                        if (i < 0 || j < 0 || i>=data_size_x || j >= data_size_y){
                            value = 0; //out of bound because of padding
                        } else {
                            value = data[d][i][j];
                        }
                        
                        sum += value * kernel.weights[d][i - responseX][j - responseY];
                        
                    }
                }
            }
            new_data[x][y] = sigmoid( sum + kernel.bias);
            
        }
    }
    return new_data;
}


float **** dataTransition(float **** dataPrevious, int n_instances, int depth, int size, Kernel *kernels, int n_kernels, Kernel poolingKernel){
    float**** newData = malloc(sizeof(float***) * n_instances);
    float**** temp = malloc(sizeof(float***) * n_instances);;
    int sizeAfterNormalConvolution = sizeAfterConvolution(size, kernels[0]); //assume that all kernels lead to same size

    for (int  i = 0; i < n_instances; i++){
        newData[i] = malloc(sizeof(float **) * n_kernels);
    }

    #pragma omp parallel for
    for (int  i = 0; i < n_instances; i++){

        #pragma omp parallel for
        for (int j = 0; j < n_kernels;j++){
            newData[i][j] = applyConvolutionWeighted(dataPrevious[i],size,size,kernels[j]);
        }

        if (poolingKernel.size != 1){
            temp[i] = newData[i];
            newData[i] = applyMaxPooling(temp[i],sizeAfterNormalConvolution,poolingKernel);
            freeImagesContinuos(temp[i],n_kernels,sizeAfterNormalConvolution);
        }
    }
    free(temp);

    return newData;
}