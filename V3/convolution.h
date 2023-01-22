
#define PROMISING_KERNEL_MEAN_MIN 0.005
#define PROMISING_KERNEL_MEAN_MAX 0.3 //should not be super high, because of max pooling

float sigmoid(float x){
    return 1 / (1 + exp(- 10 *x));
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


int sizeAfterConvolution(int originalSize, Kernel kernel){
    if (kernel.padding){
        originalSize += 2 * (kernel.size-1);
    }

    return (originalSize - kernel.size + 1) / kernel.stride ;
}

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

Kernel copyKernel(Kernel k){
    Kernel copy = createKernel(k.size,k.depth,k.type,k.stride,k.padding);
    if (k.type ==  weighted){
        for (int i = 0; i < k.depth; i++){
            for (int j = 0; j < k.size; j ++){
                for (int l = 0; l < k.size; l ++){
                    copy.weights[i][j][l] = k.weights[i][j][l]; 
                }
            }
        }
        copy.bias = k.bias;
    }
    return copy;
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
                        printf("%.3f ",kernel.weights[k][i][j]);
                    }else {
                        printf(" %.3f ",kernel.weights[k][i][j]);
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

float ** applyConvolutionWeighted(float*** data, int data_size_x, int data_size_y, Kernel kernel, bool apply_sigmoid){

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

            new_data[x][y] = sum + kernel.bias;
            if (apply_sigmoid){
                new_data[x][y] = sigmoid(new_data[x][y]);
            } 
        }
    }
    return new_data;
}

//with replacement!
//TODO: do not reallocate subset data every time!
void dataTransitionSubset(float **** dataPrevious, int n_data, int depth, int size, Kernel *kernels, int n_kernels, Kernel poolingKernel, int subset_size, float **** subset_data,  int * labels, int * subset_labels){
    float*** temp ;
    int sizeAfterNormalConvolution = sizeAfterConvolution(size, kernels[0]);
    int random_index;

    #pragma omp parallel for private(random_index, temp)
    for (int  i = 0; i < subset_size; i++){
        subset_data[i] = malloc(sizeof(float **) * n_kernels);
        random_index = rand() % n_data;
        subset_labels[i] = labels[random_index];

        #pragma omp parallel for
        for (int j = 0; j < n_kernels;j++){
            subset_data[i][j] = applyConvolutionWeighted(dataPrevious[random_index],size,size,kernels[j],true);
        }

        if (poolingKernel.size != 1){
            temp = subset_data[i];
            subset_data[i] = applyMaxPooling(temp,sizeAfterNormalConvolution,poolingKernel);
            freeImagesContinuos(temp,n_kernels,sizeAfterNormalConvolution);
        }
    }
}

float **** dataTransition(float **** dataPrevious, int n_data, int depth, int size, Kernel *kernels, int n_kernels, Kernel poolingKernel){
    float**** newData = malloc(sizeof(float***) * n_data);
    float*** temp;
    int sizeAfterNormalConvolution = sizeAfterConvolution(size, kernels[0]); //assume that all kernels lead to same size


    #pragma omp parallel for private(temp)
    for (int  i = 0; i < n_data; i++){
        newData[i] = malloc(sizeof(float **) * n_kernels);

        #pragma omp parallel for
        for (int j = 0; j < n_kernels;j++){
            newData[i][j] = applyConvolutionWeighted(dataPrevious[i],size,size,kernels[j],true);
        }

        if (poolingKernel.size != 1){
            temp = newData[i];
            newData[i] = applyMaxPooling(temp,sizeAfterNormalConvolution,poolingKernel);
            freeImagesContinuos(temp,n_kernels,sizeAfterNormalConvolution);
        }
    }
    return newData;
}

float float_rand( float min, float max ){
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

//creates a kernel that has at least X % of pixels above and below 0.5
Kernel createPromisingKernel(int size, int depth, int stride, bool padding, float **** testImages, int test_data_size, int n_test_images, bool verbose){
    float max_bias = 20, min_bias = -20, mean = 0;
    float ** kernel_response;
    Kernel k = createKernel(size,depth,weighted,stride,padding);
    int iteration  = 0, size_after = sizeAfterConvolution(test_data_size,k);

    while ( (mean < PROMISING_KERNEL_MEAN_MIN) || (PROMISING_KERNEL_MEAN_MAX < mean)){
        k.bias = float_rand(min_bias,max_bias);
        if (verbose) printf("CREAT_PROMISING_KERNEL: Iteration %d, testing out bias %f\n",iteration,k.bias);
        mean  = 0 ;
        for (int i = 0; i < n_test_images; i++){
            kernel_response = applyConvolutionWeighted(testImages[i],test_data_size,test_data_size,k,true);
            for (int x = 0; x < size_after; x++){
                for (int y = 0; y < size_after; y++){
                    if (0.5 < kernel_response[x][y]){
                        mean += 1.0;
                    }
                }
            }
            freeImageContinuos(kernel_response,size_after);
        }
        mean /= (float) (size_after * size_after * n_test_images);
        if (mean < PROMISING_KERNEL_MEAN_MIN){
            min_bias = k.bias;
        }
        if (PROMISING_KERNEL_MEAN_MAX < mean){
            max_bias = k.bias;
        }

        if (verbose) printf("\t mean is %f\n",mean);
        iteration += 1;
        if (iteration > 20){
            if (verbose) printf("RESET WEIGHTS\n");
            freeKernel(k);
            k = createKernel(size,depth,weighted,stride,padding);
            iteration = 0;
            max_bias = 20;
            min_bias = -20;
        }
    }
    return k;
}

float proportionWhite(float **** data, int n_data, int depth, int size){
    float proportion = 0.0;
    for (int i = 0; i< n_data; i++){
        for (int d = 0; d < depth; d++){
            for (int x = 0; x < size; x++){
                for (int y  = 0; y < size; y++){
                    if (0.5 < data[i][d][x][y]){
                        proportion +=1;
                    }
                }
            }
        }
    }
    return proportion / (n_data * depth * size * size);
}

float proportionWhiteLayer(float **** data, int n_data, int layer, int size){
    float proportion = 0.0;
    for (int i = 0; i< n_data; i++){
        for (int x = 0; x < size; x++){
            for (int y  = 0; y < size; y++){
                if (0.5 < data[i][layer][x][y]){
                    proportion +=1;
                }
            }
        }
    }
    return proportion / (n_data * size * size);
}
