

typedef enum KernelType {
    pooling, mustTMustFEither, weighted
}KernelType ;

typedef enum MT_MF_E_Value {
    must_true, must_false, either
}MT_MF_E_Value;

typedef struct Kernel {
    int size;
    KernelType type;
    int stride;
    bool padding;
    int depth;

    //only used for MustTMustFEither Kernels:
    MT_MF_E_Value*** map;

    //only used for Weighted Kernels:
    float bias;
    float *** weights; 
  
}Kernel;

//initializes a kernel with random weights/map
Kernel createKernel(int size,int depth, KernelType type, int stride, bool padding){

    Kernel kernel;
    kernel.size = size;
    kernel.type = type;
    kernel.stride = stride;
    kernel.padding = padding;
    kernel.depth = depth;

    switch (type){
        case mustTMustFEither:
            kernel.map = malloc(sizeof(MT_MF_E_Value**) * depth);
            for (int i = 0; i < depth; i++){
                kernel.map[i] = malloc(sizeof(MT_MF_E_Value*) * size);
                for (int j = 0; j < size; j ++){
                    kernel.map[i][j] = malloc(sizeof(MT_MF_E_Value) * size);
                    for (int k = 0; k < size; k ++){
                        int randomNumber = rand() % 3;
                        if (randomNumber == 0){
                            kernel.map[i][j][k] = must_true;
                        } else if (randomNumber == 1){
                            kernel.map[i][j][k] = must_false;
                        } else {
                            kernel.map[i][j][k] = either;
                        }
                    }
                }
            }
            break;
        case weighted:
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
            break;

    }
    return kernel;
}

void printKernel(Kernel kernel){
    if (kernel.type == pooling){
        printf("Pooling kernel with size %d\n",kernel.size);
    }else if (kernel.type == mustTMustFEither){
        printf("mustTMustFEither kernel with size %d:\n",kernel.size);
        for (int k = 0; k < kernel.depth; k++){
            printf("Depth %d:\n", k);
            for (int i = 0; i < kernel.size; i++){
                for (int j = 0; j < kernel.size; j ++){
                    if (kernel.map[k][i][j] == must_true){
                        printf("T ");
                    }else if (kernel.map[k][i][j] == must_false){
                        printf("F ");
                    }else{
                        printf("E ");
                    }
                }
                printf("\n");
            }
        }
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
    switch (kernel.type){
        case mustTMustFEither:
            for (int i = 0; i < kernel.depth; i++){
                for (int j = 0; j < kernel.size; j++ ){
                    free(kernel.map[i][j]);
                }
                free(kernel.map[i]);
            }
            free(kernel.map);
            break;
        case weighted:
            for (int i = 0; i < kernel.depth; i++){
                for (int j = 0; j < kernel.size; j++ ){
                    free(kernel.weights[i][j]);
                }
                free(kernel.weights[i]);
            }
            free(kernel.weights);
            break;
    }
}

int sizeAfterConvolution(int originalSize, Kernel kernel){
    if (kernel.padding){
        originalSize += 2 * (kernel.size-1);
    }
    int convolutions = 0;
    for (int i = 0; i  + kernel.size  <= originalSize; i+= kernel.stride){
        convolutions++;
    }
    return convolutions;
}


bool *** applyMaxPooling(bool ***data, int data_size, Kernel kernel){
    if (kernel.type != pooling){
        printf("wrong type of kernel for pooling");
    }

    int newSize = sizeAfterConvolution(data_size,kernel);
    int d,x,y,i,j,responseX,responseY;
    bool isFalse;

    bool *** newData = malloc(sizeof(bool **) * kernel.depth);
    for (i = 0; i < kernel.depth; i++){
        newData[i] = malloc(sizeof(bool*) * newSize);
        for (j = 0; j < newSize; j++){
            newData[i][j] = malloc(sizeof(bool) * newSize);
        }
    }

    for (d = 0; d < kernel.depth; d++){
        for (x = 0; x < newSize; x++){
            for (y = 0; y < newSize; y++){

                responseX = x * kernel.stride - (kernel.padding ? kernel.size -1: 0);
                responseY = y * kernel.stride - (kernel.padding ? kernel.size -1: 0);

                bool isFalse = false;
                for (i = responseX; i < responseX +  kernel.size; i++){
                    for (j = responseY; j < responseY + kernel.size; j++){

                        if (i < 0 || j < 0 || i>=data_size || j >= data_size){
                            continue;; //out of bound because of padding
                        } else {
                            isFalse = isFalse || data[d][i][j];
                        }
                    }
                }
                newData[d][x][y] = isFalse;
            }
        }
    }
    return newData;
}


//applies convolution with weighted or mustTmustFE kernel
bool ** applyConvolution(bool*** data, int data_size, Kernel kernel){

    int responseX, responseY,x,y,i,j,d;
    int newSize = sizeAfterConvolution(data_size,kernel);
    bool value, resultsTrue;
    float sum;

    bool** new_data = malloc(sizeof(bool*) * newSize);
    for (int i = 0; i < newSize; i++){
        new_data[i] = malloc(sizeof(bool) * newSize);
    }

    for (x = 0; x < newSize; x++){
        for (y = 0; y < newSize; y++){

            responseX = x * kernel.stride - (kernel.padding ? kernel.size -1: 0);
            responseY = y * kernel.stride - (kernel.padding ? kernel.size -1: 0);

            //mustTMustFEither : 
            if (kernel.type == mustTMustFEither){
                resultsTrue = true;
                for (d = 0; d < kernel.depth; d++){
                    for (i = responseX; i < responseX +  kernel.size; i++){
                        for (j = responseY; j < responseY + kernel.size; j++){

                            if (i < 0 || j < 0 || i>=data_size || j >= data_size){
                                value = false; //out of bound because of padding
                            } else {
                                value = data[d][i][j];
                            }
                            MT_MF_E_Value kernelValue = kernel.map[d][i - responseX][j - responseY];
                            if (kernelValue == must_true && !value){
                                resultsTrue = false;
                            }else if (kernelValue == must_false && value){
                                resultsTrue = false;
                            }
                            if (! resultsTrue ){
                                break;
                            }
                        }
                        if (! resultsTrue ){
                            break;
                        }
                    }
                    if (! resultsTrue ){
                        break;
                    }
                }
                new_data[x][y] = resultsTrue;
            } else {
                sum = 0;
                for (d = 0; d < kernel.depth; d++){
                    for (i = responseX; i < responseX +  kernel.size; i++){
                        for (j = responseY; j < responseY + kernel.size; j++ ){
                            if (i < 0 || j < 0 || i>=data_size || j >= data_size){
                                value = false; //out of bound because of padding
                            } else {
                                value = data[d][i][j];
                            }
                            if (value){
                                sum += kernel.weights[d][i - responseX][j - responseY];
                            }
                        }
                    }
                }
                new_data[x][y] = sum >= kernel.bias;
            }
        }
    }
    printf("END\n");
    return new_data;
}
