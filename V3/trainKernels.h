

#define GRADIENT_DESCENT_MOMENTUM = 0.5
#define GRADIENT_DESCENT_ = 0.5

void initKernels(ConvolutionalBayesianNetwork cbn, int layer, float ***test_images, int n_test_images, bool verbose){

    int d = 1,s = 28;
    float ****layered_images, **** temp;
    layered_images = imagesToLayeredImagesContinuos(test_images,n_test_images,28);
    for (int l = 0; l < layer; l++){
        temp = layered_images;
        layered_images = dataTransition(temp,n_test_images,d,s,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

        freeLayeredImagesContinuos(temp,n_test_images,d,s);

        d = cbn->bayesianNetworks[l]->depth;
        s = cbn->bayesianNetworks[l]->size;
    }

    if (verbose) printf("INIT_KERNELS: data init done\n");

    Kernel k,k_old;
    for (int i = 0; i < cbn->n_kernels[layer]; i++){
        k_old = cbn->transitionalKernels[layer][i];

        if (verbose) printf("INIT_KERNELS: create promising kernel %d of %d\n",i,cbn->n_kernels[layer]);

        cbn->transitionalKernels[layer][i] = createPromisingKernel(k_old.size,k_old.depth,k_old.stride,k_old.padding, layered_images,s,n_test_images,verbose);

        freeKernel(k_old);
    }
    
    freeLayeredImagesContinuos(layered_images,n_test_images,d,s);
    if (verbose) printf("INIT_KERNELS: Done\n");
}

float **** initGradient(int n_kernels, int kernel_depth, int kernel_size){
    float **** gradient = malloc(sizeof(float***) * n_kernels);
    for (int i = 0; i < n_kernels; i++){
        gradient[i] = malloc(sizeof(float**) * kernel_depth);
        for (int d = 0; d < kernel_depth; d++){
            gradient[i][d] = malloc(sizeof(float *) * kernel_size);
            for (int x  =0; x < kernel_size; x++){
                gradient[i][d][x] = malloc(sizeof(float) * kernel_size);
                for (int y  =0; y < kernel_size; y++){
                    gradient[i][d][x][y] = 0;
                }
            }
        }
    }
    return gradient;
}

void resetGradient(float **** gradient, int n_kernels, int kernel_depth, int kernel_size){
    for (int i = 0; i < n_kernels; i++){
        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){
                    gradient[i][d][x][y] = 0;
                }
            }
        }
    }
}

void multiplyGradient(float **** gradient, float x, int n_kernels, int kernel_depth, int kernel_size){
    for (int i = 0; i < n_kernels; i++){
        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){
                    gradient[i][d][x][y] *= x;
                }
            }
        }
    }
}

void freeGradient(float **** gradient, int n_kernels, int kernel_depth, int kernel_size){
    for (int i = 0; i < n_kernels; i++){
        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                free(gradient[i][d][x]);
            }
            free(gradient[i][d]);
        }
        free(gradient[i]);
    }
    free(gradient);
}

//sets g1 to "weights * g1 + (1 - weights) * g2"
void additionGradients(float **** g1, float **** g2, float weight , int n_kernels, int kernel_depth, int kernel_size){
    for (int i = 0; i < n_kernels; i++){
        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){
                    g1[i][d][x][y] =  weight * g1[i][d][x][y] + (1 - weight) * g2[i][d][x][y];
                }
            }
        }
    }
}

void updateKernels(Kernel * kernels, int n_kernels, int kernel_depth, int kernel_size, float **** gradient){
    for (int i = 0; i < n_kernels; i++){
        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){
                    kernels[i].weights[d][x][y] += gradient[i][d][x][y];
                }
            }
        }
    }
}

/* 
bool * parent_states = malloc(sizeof(bool) * nn->n_parents);
double result;
for (int i = 0; i < nn->n_parents; i++){
    parent_states[i] = nn->parents[i]->value;
}
result = nn->CPT[binaryToInt(parent_states,nn->n_parents)][nn->value];
free(parent_states);
 */

float prob_nn_given_data_dependent_n(NumberNode nn, Node n, bool n_value, int number_label, float *** data){
    Node parent;
    float prob;
    int counts_row, row_number;
    bool * parent_states = malloc(sizeof(bool) * nn->n_parents);
    for (int j = 0; j < nn->n_parents; j++){
        parent = nn->parents[j];
        if (parent->x == n->x && parent->y == n->y && parent->depth == n->depth){
            parent_states[j] = n_value;
        } else {
            parent_states[j]  = 0.5 < data[parent->depth][parent->x][parent->y];
        }
    }
    row_number = binaryToInt(parent_states,nn->n_parents);
    counts_row = 0;
    for (int j = 0; j < 10; j++){
        counts_row += nn->stateCounts[counts_row][j];
    }

    prob = (float)(nn->stateCounts[counts_row][number_label]) / (float)(counts_row);
    free(parent_states);
    return prob;
}

//assumes that nn is set to the correct state already
void calculateGradientNumberNodeChildren(float *** gradient, Node n, Kernel kernel, int n_kernels, Kernel poolingKernel, float *** data_before
    ,int before_depth, int before_size, float *** data_intermediate_before_sigmoid, int size_intermediate, float *** data_after, int number_label){
    
    NumberNode nn;
    int x_pooling, y_pooling, responseX, responseY;
    float prob_if_true, prob_if_false, a, value, max_value;
    for (int i = 0; i < n->n_numberNodeChildren; i++){
        nn = n->numberNodeChildren[i];
        prob_if_true = prob_nn_given_data_dependent_n(nn,n,true,number_label,data_after);
        prob_if_false = prob_nn_given_data_dependent_n(nn,n,false,number_label,data_after);

        //how to change in order to increase probability
        a = prob_if_true  - prob_if_false;

        //find node position responsible for value via pooling
        responseX = n->x * poolingKernel.stride - (poolingKernel.padding ? poolingKernel.size -1: 0);
        responseY = n->y * poolingKernel.stride - (poolingKernel.padding ? poolingKernel.size -1: 0);
        max_value = -999999999;
        for (int x = responseX; x < responseX +  poolingKernel.size; x++){
            for (int y = responseY; y < responseY + poolingKernel.size; y++ ){
                if (x < 0 || y < 0 || x >= size_intermediate || y >= size_intermediate){
                    continue;; //out of bound because of padding
                }
                value = data_intermediate_before_sigmoid[n->depth][x][y];
                if (max_value < value){ //alreanativly (value == sigmoid(value) == data_after[n->depth][x][y]) shoudl work as well
                    max_value = value;
                    x_pooling = x;
                    y_pooling = y;
                }
                
            }
        }

        //multiply by sigmoid derivative
        a *= sigmoid(data_intermediate_before_sigmoid[n->depth][x_pooling][y_pooling]) 
            * (1 - data_intermediate_before_sigmoid[n->depth][x_pooling][y_pooling]);

        //add to gradient
        responseX = x_pooling * kernel.stride - (kernel.padding ? kernel.size -1: 0);
        responseY = y_pooling  * kernel.stride - (kernel.padding ? kernel.size -1: 0);
        for (int d = 0; d < before_depth; d++){
            for (int x = responseX; x < responseX +  kernel.size; x++){
                for (int y = responseY; y < responseY + kernel.size; y++ ){

                    //check if out of bound (could be, because of padding)
                    if (0 < x && x < before_size && 0 < y && y < before_size ){
                        //dependent on how high the value based on it actually is
                        gradient[d][x][y] += data_before[d][x][y] * a;
                    }
                    
                }
            }
        }
    }
}

void calculateGradientImage(BayesianNetwork bn, Kernel * kernels, int n_kernels, Kernel poolingKernel,float *** representation_before, int n_data, float nodeFraction ){
    //nothing for now
}

void calculateGradient(BayesianNetwork bn, float **** gradient, int n_kernels, float **** data_before, int n_data, Kernel * kernels, Kernel poolingKernel , float learning_rate, int batchSize, float batchNodeFraction){
    int kernel_depth = kernels[0].depth;
    int kernel_size = kernels[0].size;
    resetGradient(gradient, n_kernels,kernel_depth,kernel_size);

    float *** example;
    for (int i = 0; i < batchSize; i++){
        example = data_before[rand() % n_data];
        
        calculateGradientImage(bn,kernels, n_kernels,poolingKernel,data_before[i],n_data,batchNodeFraction);

    }


    multiplyGradient(gradient, learning_rate * (1.0 / batchSize), n_kernels,kernel_depth,kernel_size);
}

void trainKernelsGradientDescent(ConvolutionalBayesianNetwork cbn, int layer, int iterations, float learning_rate ,float momentum, int batchSize, float batchNodeFraction
    , float *** images, int * labels, int n_data, int n_data_used_for_counts, bool verbose){

    if (verbose) printf("GRAD_DEC: start init\n");

    int n_kernels = cbn->n_kernels[layer];
    int kernel_depth = cbn->transitionalKernels[layer][0].depth;
    int kernel_size = cbn->transitionalKernels[layer][0].size;
    Kernel * kernels = cbn->transitionalKernels[layer];
    Kernel pooling_kernel  =cbn->poolingKernels[layer];
    BayesianNetwork bn = cbn->bayesianNetworks[layer];

    float **** gradient = initGradient(n_kernels,kernel_size,kernel_depth);
    float **** previousGradient = initGradient(n_kernels,kernel_size,kernel_depth);

    float ****temp, **** data_previous_layer = imagesToLayeredImagesContinuos(images,n_data,28);
    float **** subset_data = malloc(sizeof(float ***) * n_data_used_for_counts);
    int * subset_labels = malloc(sizeof(int) * n_data_used_for_counts);
    int d = 1,s = 28;
    for (int l = 0; l < layer; l++){
        temp = data_previous_layer;
        data_previous_layer = dataTransition(temp,n_data,d,s
            ,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

        freeLayeredImagesContinuos(temp,n_data,d,s);
        d = cbn->bayesianNetworks[l]->depth;
        s = cbn->bayesianNetworks[l]->size;
    }

    if (verbose) printf("GRAD_DEC: init done\n");

    for (int i = 0; i< iterations; i++){

        if (verbose) printf("GRAD_DEC: iteration %d of %d\n",i,iterations);

        if (verbose) printf("GRAD_DEC: learn gradien\n");
        calculateGradient(bn, gradient,n_kernels,data_previous_layer,n_data,kernels,pooling_kernel,learning_rate,batchSize,batchNodeFraction);
        
        if (verbose) printf("GRAD_DEC: combine running gradient\n");
        additionGradients(previousGradient,gradient,momentum,n_kernels,kernel_depth,kernel_size);

        if (verbose) printf("GRAD_DEC: update kernels\n");
        updateKernels(kernels,n_kernels,kernel_depth,kernel_size,previousGradient);

        if (verbose) printf("GRAD_DEC: transform subset data\n");
        dataTransitionSubset(data_previous_layer,n_data,d,s,kernels,n_kernels,pooling_kernel,n_data_used_for_counts,subset_data,labels, subset_labels);

        if (verbose) printf("GRAD_DEC: fit data counts n\n");
        fitDataCounts(bn,subset_data,n_data_used_for_counts);

        if (verbose) printf("GRAD_DEC: fit data counts nn\n");
        for (int i = 0; i < bn->n_numberNodes; i++){
            fitDataCountsNumberNode(bn->numberNodes[i],subset_data,subset_labels, n_data_used_for_counts);
        }

        if (verbose) printf("GRAD_DEC: free subset\n");
        for (int i = 0; i < n_data_used_for_counts; i++){
            freeImagesContinuos(subset_data[i],bn->depth,bn->size);
        }
    }

    if (verbose) printf("GRAD_DEC: done -> free everything\n");

    float **** all_data =  dataTransition(data_previous_layer,n_data,d,s,kernels,n_kernels, pooling_kernel);
    //finally update ALL counts
    fitDataCounts(bn, all_data, n_data);

    freeLayeredImagesContinuos(data_previous_layer,n_data,d,s);
    d = bn->depth;
    s = bn->size;
    freeLayeredImagesContinuos(all_data,n_data,bn->depth,bn->size);
    free(subset_data);
    free(subset_labels);
    freeGradient(gradient,n_kernels,kernel_depth,kernel_size);
    freeGradient(previousGradient,n_kernels,kernel_depth,kernel_size);
    if (verbose) printf("GRAD_DEC: exit\n");
}