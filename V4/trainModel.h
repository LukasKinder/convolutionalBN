
#define N_TEST_PROPORTION_WHITE 30
#define NORMALIZE_KERNELS_INDIVIDUALLY true

typedef struct RawGradient *Gradient;

typedef struct RawGradient {

    int n_kernels;
    int kernel_depths;
    int kernel_sizes;
    float **** kernel_gradients; //n_kernels * kernel_depth * kernel_sizes * kernel_sizes

    float *bias_gradients; //n_kernels


    int node_depth;
    int node_size;
    int node_neighbors_size;
    float ****** perceptron_gradients; // node_depth * node_size * node_size * node_depth * node_neighbors_size * node_neighbors_size

    float learning_rate;

    int gaussian_blur_size;
    float gaussian_blur_sd;
    float ** gaussian_blur_map;

}RawGradient;

Gradient initGradient(int n_kernels, int kernel_depth, int kernel_sizes, int node_depth, int node_size, int node_neighbour_size, float learning_rate
        ,int gaussian_blur_size, float gaussian_blur_sd){
    
    if (gaussian_blur_size %2 == 0){
        printf("error, map for gaussian blur should not have an even size\n");
        exit(1);
    }

    int i,j,k,d,x,y;

    Gradient g = malloc(sizeof(RawGradient));

    g->n_kernels = n_kernels;
    g->kernel_depths = kernel_depth;
    g->kernel_sizes = kernel_sizes;
    g->kernel_gradients = malloc(sizeof(float***) * g->n_kernels);
    for (i = 0; i < g->n_kernels; i++){
        g->kernel_gradients[i] = malloc(sizeof(float **) * g->kernel_depths);
        for (j = 0; j < g->kernel_depths; j++){
            g->kernel_gradients[i][j] = malloc(sizeof(float*) * kernel_sizes);
            for (k = 0; k < g->kernel_sizes; k++){
                g->kernel_gradients[i][j][k] = malloc(sizeof(float) * kernel_sizes);
            }
        }
    }

    g->bias_gradients = malloc(sizeof(float) * n_kernels);

    g->node_depth = node_depth;
    g->node_size = node_size;
    g->node_neighbors_size = node_neighbour_size;

    //float ****** perceptron_gradients; // node_depth * node_size * node_size * node_depth * node_neighbors_size * node_neighbors_size
    g->perceptron_gradients = malloc(sizeof(float*****) * g->node_depth);
    for (d = 0; d < g->node_depth; d++){
        g->perceptron_gradients[d] = malloc(sizeof(float****) * g->node_size);
        for (x = 0; x < g->node_size; x++){
            g->perceptron_gradients[d][x] = malloc(sizeof(float***) * g->node_size);
            for (y = 0; y < g->node_size; y++){
                g->perceptron_gradients[d][x][y] = malloc(sizeof(float**) * g->node_depth);
                for (i = 0; i < g->node_depth; i++){
                    g->perceptron_gradients[d][x][y][i] = malloc(sizeof(float*) * g->node_neighbors_size);
                    for (j  =0; j < g->node_neighbors_size; j++){
                        g->perceptron_gradients[d][x][y][i][j] = malloc(sizeof(float) * g->node_neighbors_size);
                    }
                }
            }
        }
    }

    g->learning_rate = learning_rate;
    g->gaussian_blur_size = gaussian_blur_size;
    g->gaussian_blur_sd = gaussian_blur_sd;
    g->gaussian_blur_map = malloc(sizeof(float) * g->gaussian_blur_size);

    float sum = 0.0;
    float d_x,d_y;
    int middle = g->gaussian_blur_size / 2;
    for (i = 0; i < g->gaussian_blur_size; i++){
        g->gaussian_blur_map[i] = malloc(sizeof(float) * g->gaussian_blur_size);
        for (int j = 0; j < g->gaussian_blur_size; j++){
            d_x = (float)(abs(middle - i);
            d_y = (float)(abs(middle - j);
            g->gaussian_blur_map[i][j] = exp(- (d_x*d_x * d_y * d_y) / 2 * (pow(g->gaussian_blur_sd,2)));
            sum += g->gaussian_blur_map[i][j];
        }
    }

    for (i = 0; i < g->gaussian_blur_size; i++){
        for (int j = 0; j < g->gaussian_blur_size; j++){
            g->gaussian_blur_map[i][j] /= sum;
        }
    }
    resetGradient(g);
    return g;
}

//sets gradient to zero
void resetGradient(Gradient g){
    
    int d,x,y,i,j,k,l;

    for (i = 0; i < g->n_kernels; i++){
        g->bias_gradients[i] = 0.0;
        g->kernel_gradients[i] = malloc(sizeof(float **) * g->kernel_depths);
        for (j = 0; j < g->kernel_depths; j++){
            g->kernel_gradients[i][j] = malloc(sizeof(float*) * kernel_sizes);
            for (k = 0; k < g->kernel_sizes; k++){
                g->kernel_gradients[i][j][k] = malloc(sizeof(float) * kernel_sizes);
                for(l = 0; l < g->kernel_sizes;l++){
                   g->kernel_gradients[i][j][k][l] = 0.0; 
                }
            }
        }
    }

    for (d = 0; d < g->node_depth; d++){
        for (x = 0; x < g->node_size; x++){
            for (y = 0; y < g->node_size; y++){
                for (i = 0; i < g->node_depth; i++){
                    for (j  =0; j < g->node_neighbors_size; j++){
                        for (k = 0; k < g->node_neighbors_size; k++){
                            g->perceptron_gradients[d][x][y][i][j][k] = 0.0;
                        }
                    }
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




void updateKernels(Kernel * kernels, int n_kernels, int kernel_depth, int kernel_size, float **** gradient, float * bias_gradient){

    for (int i = 0; i < n_kernels; i++){
        kernels[i].bias += bias_gradient[i];
        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){
                    kernels[i].weights[d][x][y] += gradient[i][d][x][y];
                }
            }
        }
    }
}


float desireToChangeNode(Node n, float *** data){

    float a = 0.0;
    Node child;
    int counts_child_true_if_n_true, counts_child_false_if_n_true;
    int counts_child_true_if_n_false, counts_child_false_if_n_false;
    int counts_child_total;

    float prob_state_child;
    float prob_child_dependent_state_n;

    for(int i = 0; i < n->n_children; i++){

        child = n->children[i];
        counts_child_given_data_dependent_n(child,n,true,data, &counts_child_true_if_n_true, &counts_child_false_if_n_true);
        counts_child_given_data_dependent_n(child,n,false,data, &counts_child_true_if_n_false, &counts_child_false_if_n_false);

        counts_child_total = counts_child_true_if_n_true + counts_child_true_if_n_false + counts_child_false_if_n_true + counts_child_false_if_n_false;
        if (counts_child_total == 0){
            continue;
        }

        //printNode(child,true);

        //printf("ctnt %d, cfnt %d ctnf %d cfnf %d\n",counts_child_true_if_n_true,counts_child_false_if_n_true, counts_child_true_if_n_false,counts_child_false_if_n_false);


        if (data[child->depth][child->x][child->y]){
            //printf("child is true\n");
            prob_state_child = (float)(counts_child_true_if_n_true + counts_child_true_if_n_false) / (counts_child_total);

            if (counts_child_true_if_n_true + counts_child_false_if_n_true == 0){
                prob_child_dependent_state_n = 1; //0.5;
            } else{
                prob_child_dependent_state_n = (float)(counts_child_true_if_n_true) / (counts_child_true_if_n_true + counts_child_false_if_n_true);
            }

            if (counts_child_true_if_n_false + counts_child_false_if_n_false == 0){
                prob_child_dependent_state_n -= 1; //0.5;
            }else{
                prob_child_dependent_state_n -= (float)(counts_child_true_if_n_false) / (counts_child_true_if_n_false + counts_child_false_if_n_false);
            }


        }else{
            //printf("child is false\n");
            prob_state_child = (float)(counts_child_false_if_n_true + counts_child_false_if_n_false) / (counts_child_total);

            if (counts_child_true_if_n_true + counts_child_false_if_n_true == 0){
                prob_child_dependent_state_n = 0.5;
            }else{
                prob_child_dependent_state_n = (float)(counts_child_false_if_n_true) / (counts_child_true_if_n_true + counts_child_false_if_n_true);
            }

            if (counts_child_true_if_n_false + counts_child_false_if_n_false == 0.5){
                prob_child_dependent_state_n -= 0.5;
            }else{
                prob_child_dependent_state_n -= (float)(counts_child_false_if_n_false) / (counts_child_true_if_n_false + counts_child_false_if_n_false);
            }
        }


        a += (1 - prob_state_child) * prob_child_dependent_state_n;

    }
    return a;
}

//assumes that nn is set to the correct state already
void calculateGradientNode(float *** gradient, float * bias_gradient, Node n, Kernel kernel, Kernel poolingKernel, float *** data_before
    ,int before_depth, int before_size, float *** data_intermediate_before_sigmoid, int size_intermediate, float *** data_after, int number_label){

    if (n->n_children == 0) return;
    
    NumberNode nn;
    int x_pooling, y_pooling, responseX, responseY;
    float prob_if_true, prob_if_false, a,b, value, max_value,s;

    a = desireToChangeNode(n,data_after);

    if (a == 0.0){
        return;
    } 

    a /= ( n->n_children);

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
    s = sigmoid(data_intermediate_before_sigmoid[n->depth][x_pooling][y_pooling], 8);

    a *= s * (1 - s);

    //add to bias
    *bias_gradient += a;

    //printf("add to gradeien:\n");
    //add to gradient
    responseX = x_pooling * kernel.stride - (kernel.padding ? kernel.size -1: 0);
    responseY = y_pooling  * kernel.stride - (kernel.padding ? kernel.size -1: 0);
    for (int d = 0; d < before_depth; d++){
        for (int x = responseX; x < responseX +  kernel.size; x++){
            for (int y = responseY; y < responseY + kernel.size; y++ ){

                //check if out of bound (could be, because of padding)
                if (0 < x && x < before_size && 0 < y && y < before_size ){
                    //dependent on how high the value based on it actually is
                    //printf("%f->%f  ",data_before[d][x][y],data_before[d][x][y] * a);
                    gradient[d][x - responseX][y - responseY] += data_before[d][x][y] * a;
                }
            }
        }
    }
}

void calculateGradientImage(float **** gradient, float * bias_gradient ,BayesianNetwork bn, Kernel * kernels, int n_kernels, Kernel poolingKernel
        , float *** representation_before, int n_data, int size_data, int number_label){
    //nothing for now
    Kernel k;
    Node n;
    float *** responseKernels = malloc(sizeof(float**) * n_kernels);
    float *** full_transformation = malloc(sizeof(float**) * n_kernels);;
    int size_after_kernel = sizeAfterConvolution(size_data,kernels[0]);
    for (int i = 0; i < n_kernels; i++){
        responseKernels[i] = applyConvolutionWeighted(representation_before,size_data,size_data,kernels[i],false);
        full_transformation[i] = applyMaxPoolingOneLayer(responseKernels[i],size_after_kernel,size_after_kernel,poolingKernel);
        //still transform with sigmoid
        for (int x = 0; x < bn->size; x++){
            for (int y  = 0; y < bn->size; y++){
                full_transformation[i][x][y] = sigmoid(full_transformation[i][x][y],1.0);
            }
        }
    }


    for (int i = 0; i < n_kernels; i++){
        k = kernels[i];

        for (int x = 0; x < bn->size; x++){
            for (int y = 0; y < bn->size; y++){
                n = bn->nodes[i][x][y];

                calculateGradientNode(gradient[i],&(bias_gradient[i]),n,k,poolingKernel,representation_before, k.depth,size_data,responseKernels
                            ,size_after_kernel,full_transformation,number_label);
            }
        }
    }
    freeImagesContinuos(responseKernels,n_kernels,size_after_kernel);
    freeImagesContinuos(full_transformation,n_kernels,bn->size);


}

void normalizeGradient(float learning_rate,float *bias_gradient, float**** gradient, int n_kernels, int kernel_depth
        , int kernel_size, bool normalize_kernel_separate){

    
    
    float * length_array = malloc(sizeof(float) * n_kernels);
    float length = 0.0;
    

    //normalize gradient
    for (int i = 0; i < n_kernels; i++){
        length_array[i] = pow(bias_gradient[i],2);
        length += pow(bias_gradient[i],2);

        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){
                    length += pow(gradient[i][d][x][y],2);
                    length_array[i] += pow(gradient[i][d][x][y],2);
                }
            }
        }
    }

    if (length == 0.0){
        printf("ERROR: gradient = 0!\n");
        exit(1);
    }

    length = sqrt(length);
    for (int i = 0; i < n_kernels; i++){
        if (length_array[i] == 0){
            printf("WARNING! no learning for kernel %d\n",i);
            length_array[i] = 1.0;
        }else{
            length_array[i] = sqrt(length_array[i]);
        }
        
    }

    for (int i = 0; i < n_kernels; i++){
        if (normalize_kernel_separate){
            bias_gradient[i] *=  learning_rate / length_array[i];
        }else {
            bias_gradient[i] *=  learning_rate / length;
        }
        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){

                    if (normalize_kernel_separate){
                        gradient[i][d][x][y] *=  learning_rate / length_array[i];
                    }else {
                        gradient[i][d][x][y] *=  learning_rate / length;
                    }
                }
            }
        }
    }
}

void calculateGradient(BayesianNetwork bn, float **** gradient, float * bias_gradient , int n_kernels, float **** data_before
    , int n_data,int size_data, int * number_labels, Kernel * kernels, Kernel poolingKernel , float learning_rate, int batchSize){

    int kernel_depth = kernels[0].depth;
    int kernel_size = kernels[0].size;
    int random_index,l;

    resetGradient(gradient, n_kernels,kernel_depth,kernel_size);
    for (int i = 0; i < n_kernels; i++){
        bias_gradient[i] = 0.0;
    }

    int n_threads = N_THREADS;
    float ***** gradient_each_thread = malloc(sizeof(float****) * n_threads);
    float ** bias_each_thread = malloc(sizeof(float*) * n_threads);
    for (int i = 0; i < n_threads; i++){
        bias_each_thread[i] = calloc(n_kernels, sizeof(float));
        gradient_each_thread[i] = initGradient(n_kernels,kernel_depth,kernel_size);
    }



    float *** example;
    int this_thread_number;
    #pragma omp parallel for private(random_index,l,this_thread_number) num_threads(n_threads)
    for (int i = 0; i < batchSize; i++){
        this_thread_number = omp_get_thread_num();
        random_index = (rand() % (n_data / n_threads)) * this_thread_number;
        example = data_before[random_index];
        l = number_labels[random_index];
        calculateGradientImage(gradient_each_thread[this_thread_number],bias_each_thread[this_thread_number] ,bn,kernels, n_kernels,poolingKernel,example,n_data,size_data,l);
    }

    for (int nk = 0; nk < n_threads; nk++){
        for (int i = 0; i < n_kernels; i++){
            bias_gradient[i] +=  bias_each_thread[nk][i];
            for (int d = 0; d < kernel_depth; d++){
                for (int x  =0; x < kernel_size; x++){
                    for (int y  =0; y < kernel_size; y++){
                        gradient[i][d][x][y] += gradient_each_thread[nk][i][d][x][y];
                    }
                }
            }
        }
    }

    for (int i = 0; i < n_threads; i++){
        free(bias_each_thread[i]);
        freeGradient(gradient_each_thread[i],n_kernels,kernel_depth,kernel_size);
    }
    free(gradient_each_thread);
    free(bias_each_thread);

    normalizeGradient(learning_rate,bias_gradient, gradient, n_kernels,kernel_depth,kernel_size,NORMALIZE_KERNELS_INDIVIDUALLY);
}





void trainKernelsGradientDescent(ConvolutionalPerceptronModel cpm, int layer, int iterations, float learning_rate ,float momentum, int batchSize
    , float *** images, int n_data, bool verbose){

    if (verbose) printf("GRAD_DEC: start init data used for counts = %d\n", n_data_used_for_counts);

    int n_kernels = cpm->n_kernels[layer];
    int kernel_depth = cpm->transitionalKernels[layer][0].depth;
    int kernel_size = cpm->transitionalKernels[layer][0].size;
    Kernel * kernels = cpm->transitionalKernels[layer];
    Kernel pooling_kernel  = cpm->poolingKernels[layer];
    PerceptronGrid pg = cpm->perceptronGrids[layer];

    float **** gradient = initGradient(n_kernels,kernel_depth,kernel_size);
    float **** previousGradient = initGradient(n_kernels,kernel_depth,kernel_size);

    float * bias_gradient = malloc(sizeof(float) * n_kernels);
    float * previous_bias_gradient = calloc(n_kernels, sizeof(float));

    float ****temp, **** data_previous_layer = imagesToLayeredImagesContinuos(images,n_data,28);
    float **** subset_data = malloc(sizeof(float ***) * n_data_used_for_counts);

    float entropyCounts;

    int * subset_labels = malloc(sizeof(int) * n_data_used_for_counts);
    int d = 1,s = 28;
    for (int l = 0; l < layer; l++){
        temp = data_previous_layer;
        data_previous_layer = dataTransition(temp,n_data,d,s
            ,cpm->transitionalKernels[l],cpm->n_kernels[l],cpm->poolingKernels[l]);

        freeLayeredImagesContinuos(temp,n_data,d,s);
        d = cpm->bayesianNetworks[l]->depth;
        s = cpm->bayesianNetworks[l]->size;
    }

    if (verbose) printf("GRAD_DEC: init done\n");

    float * proportion_white_kernels = malloc(sizeof(float) * n_kernels);
    update_proportion_white(proportion_white_kernels,kernels,n_kernels,data_previous_layer,n_data,d,s);

    for (int it = 0; it< iterations; it++){

        if (verbose) printf("GRAD_DEC: iteration %d of %d\r",it,iterations);
        if (it % mod_update_counts == 0){
            if (verbose) printf("\nGRAD_DEC: transform subset data (subset size = %d)\n", n_data_used_for_counts);
            dataTransitionSubset(data_previous_layer,n_data,d,s,kernels,n_kernels,pooling_kernel,n_data_used_for_counts,subset_data,labels, subset_labels);

            fitDataCounts(bn,subset_data,n_data_used_for_counts); 
            for (int j = 0; j < bn->n_numberNodes; j++){
                fitDataCountsNumberNode(bn->numberNodes[j],subset_data,subset_labels, n_data_used_for_counts);
            }
            

            entropyCounts = averageEntropyCounts(bn) - averageEntropyCountsNoRelations(bn);

            if (verbose) printf("average gain entropy in CPT rows: %f \n",entropyCounts);


            if (bn->learning_curve_len == bn->learning_curve_size){
                bn->learning_curve_len *=2;
                bn->learning_curve = realloc(bn->learning_curve, sizeof(float) * bn->learning_curve_len);
                for (int i = 0; i < bn->learning_proportion_white_n_channels; i++){
                    bn->learning_proportion_white[i] = realloc(bn->learning_proportion_white[i], sizeof(float) * bn->learning_curve_len);
                }
            }
            bn->learning_curve[bn->learning_curve_size] = entropyCounts ;
            for (int i = 0; i < bn->learning_proportion_white_n_channels; i++){
                bn->learning_proportion_white[i][bn->learning_curve_size] = proportionWhiteLayer(subset_data,n_data_used_for_counts / 10,i,bn->size);
            }
            bn->learning_curve_size +=1;

            if (verbose) printf("GRAD_DEC: free subset\n");
            for (int j = 0; j < n_data_used_for_counts; j++){
                freeImagesContinuos(subset_data[j],bn->depth,bn->size);
            }
        }

        calculateGradient(bn, gradient,bias_gradient,n_kernels,data_previous_layer,n_data,s, labels, kernels,pooling_kernel,learning_rate,batchSize);

        additionGradients(previousGradient,gradient,momentum,n_kernels,kernel_depth,kernel_size);
        for (int j = 0; j < n_kernels; j++){
            previous_bias_gradient[j] = momentum * previous_bias_gradient[j] + (1 - momentum) * bias_gradient[j];
        }
     

        updateKernels(kernels,n_kernels,kernel_depth,kernel_size,previousGradient, previous_bias_gradient);

        moderateBiases(proportion_white_kernels,kernels,n_kernels,data_previous_layer,n_data,d,s);
    }

    if (verbose) printf("\nGRAD_DEC: done -> free everything\n");
    freeLayeredImagesContinuos(data_previous_layer,n_data,d,s);
    d = bn->depth;
    s = bn->size;
    free(subset_data);
    free(subset_labels);
    freeGradient(gradient,n_kernels,kernel_depth,kernel_size);
    freeGradient(previousGradient,n_kernels,kernel_depth,kernel_size);
    free(bias_gradient);
    free(previous_bias_gradient);
    free(proportion_white_kernels);
    if (verbose) printf("GRAD_DEC: exit\n");
}