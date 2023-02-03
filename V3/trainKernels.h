
#define N_THREADS 16
#define N_TEST_PROPORTION_WHITE 30
#define NORMALIZE_KERNELS_INDIVIDUALLY true
#define IDEAL_CHANGE_PROPORTION_WHITE 0.001

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

void printGradients(float **** gradient, float * bias_gradients, int n_kernels, int kernel_depth, int kernel_size){
    printf("bias gradients:\n");
    for (int i = 0; i < n_kernels; i++){
        printf("\t %f",bias_gradients[i]);
    }
    printf("\n");
    for (int i = 0; i < n_kernels; i++){
        printf("weight gradients kernel %d:\n",i);
        for (int d = 0; d < kernel_depth; d++){
            printf("depth %d:\n",d);
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){
                    printf(" %.6f",gradient[i][d][x][y]);
                }
                printf("\n");
            }
        }
    }

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

/* 
bool * parent_states = malloc(sizeof(bool) * nn->n_parents);
double result;
for (int i = 0; i < nn->n_parents; i++){
    parent_states[i] = nn->parents[i]->value;
}
result = nn->CPT[binaryToInt(parent_states,nn->n_parents)][nn->value];
free(parent_states);
 */

float average_prob_true_n(Node n){
    int count_f = 0;
    int count_t = 0;
    int n_rows = pow(2,n->n_parents);
    for (int i  =0; i < n_rows; i++){
        count_t += n->stateCountsTrue[i];
        count_f += n->stateCountsFalse[i];
    }
    if (count_t + count_f == 0){
        return 0.5;
    }
    return (float)(count_t) / (float)(count_t + count_f);

}

float prob_n_given_data(Node n, float *** data){
    Node parent;
    int counts_row, row_number;
    bool * parent_states = malloc(sizeof(bool) * n->n_parents);

    for (int j = 0; j < n->n_parents; j++){
        parent = n->parents[j];
        parent_states[j]  = 0.5 < data[parent->depth][parent->x][parent->y];
        
    }
    row_number = binaryToInt(parent_states,n->n_parents);
    free(parent_states);

    counts_row = n->stateCountsTrue[row_number] + n->stateCountsFalse[row_number];
    if (counts_row == 0){
        return 0.5;
    }


    
    return (float)(n->stateCountsTrue[row_number]) / (float)(counts_row);
}

void counts_child_given_data_dependent_n(Node child, Node n, bool n_value, float *** data, int *counts_child_true, int *counts_child_false){

    Node parent;

    bool own_state =  0.5 < data[child->depth][child->x][child->y];
    bool * parent_states = malloc(sizeof(bool) * child->n_parents);

    for (int j = 0; j < child->n_parents; j++){
        parent = child->parents[j];
        if (parent->x == n->x && parent->y == n->y && parent->depth == n->depth){
            parent_states[j] = n_value;
        } else {
            parent_states[j]  = 0.5 < data[parent->depth][parent->x][parent->y];
        }
    }
    int row_number = binaryToInt(parent_states,child->n_parents);
    free(parent_states);

    *counts_child_true = child->stateCountsTrue[row_number];
    *counts_child_false = child->stateCountsFalse[row_number];
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

        /* printf("prob_state_child = %f\n",prob_state_child);
        printf("prob _child dependent n %f\n",prob_child_dependent_state_n); */

        /*printf("desired change child = %f\n", (1 - prob_state_child) * prob_child_dependent_state_n);

        int X;
        scanf("%d",&X); */


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

void update_proportion_white(float * proportion_white_kernels, Kernel * kernels, int n_kernels, float **** data_previous_layer, int n_data, int d , int s ){
    n_data = N_TEST_PROPORTION_WHITE < n_data ? N_TEST_PROPORTION_WHITE : n_data;

    int size_after = sizeAfterConvolution(s,kernels[0]);

    #pragma omp parallel for
    for (int i = 0; i < n_kernels; i++){
        proportion_white_kernels[i] = 0.0;

        #pragma omp parallel for
        for (int j = 0; j < n_data; j++){
            float ** data_after;
            data_after = applyConvolutionWeighted(data_previous_layer[j],s,s,kernels[i],false);
            for (int x = 0;  x < size_after; x++){
                for (int y = 0; y < size_after; y++){
                    proportion_white_kernels[i] += data_after[x][y];
                }
            }
            freeImageContinuos(data_after,size_after);
        }
        proportion_white_kernels[i] /= n_data * size_after * size_after;
    }

}

void moderateBiases(float * proportion_white_kernels, Kernel * kernels, int n_kernels, float **** data_previous_layer, int n_data, int d ,int s ){

    int size_after = sizeAfterConvolution(s,kernels[0]);

    float * new_proportion_white = malloc(sizeof(float) * n_kernels);
    update_proportion_white(new_proportion_white,kernels,n_kernels,data_previous_layer, n_data, d, s);
    bool * should_increase = malloc(sizeof(bool) * n_kernels);
    for (int i = 0; i < n_kernels; i++){
        should_increase[i] = proportion_white_kernels[i] < new_proportion_white[i] ? true : false;
    }


    

    //1, 0.5, 0.25, 0.125 ...
    float change = 10.0;
    for (int iteration = 0; iteration < 15; iteration++){

        if (iteration != 0){
            update_proportion_white(new_proportion_white,kernels,n_kernels,data_previous_layer, n_data, d, s);
        }

        for (int i = 0; i < n_kernels; i++){
            
            if (should_increase[i]){
                //should increase...
                if (new_proportion_white[i] < proportion_white_kernels[i] + IDEAL_CHANGE_PROPORTION_WHITE){
                    kernels[i].bias += change;
                } else{
                    kernels[i].bias -= change;
                }
            }else{
                //should decrease...
                if (new_proportion_white[i] < proportion_white_kernels[i] - IDEAL_CHANGE_PROPORTION_WHITE){
                    kernels[i].bias += change;
                } else {
                    kernels[i].bias -= change;
                }
            }
        }


        change /=2;
    }

    for (int i = 0; i < n_kernels; i++){
        if (fabs(proportion_white_kernels[i] - new_proportion_white[i]) > IDEAL_CHANGE_PROPORTION_WHITE * 3){
            printf("WARNING: moderating proportion white failed for kernel %d! \n change of %f\n\n",i,proportion_white_kernels[i] - new_proportion_white[i]);
        }
        proportion_white_kernels[i] = new_proportion_white[i];
    }

    free(new_proportion_white);
    free(should_increase);
}


void trainKernelsGradientDescent(ConvolutionalBayesianNetwork cbn, int layer, int iterations, int mod_update_counts, float learning_rate ,float momentum, int batchSize
    , float *** images, int * labels, int n_data, int n_data_used_for_counts, bool verbose){

    if (verbose) printf("GRAD_DEC: start init data used for counts = %d\n", n_data_used_for_counts);

    int n_kernels = cbn->n_kernels[layer];
    int kernel_depth = cbn->transitionalKernels[layer][0].depth;
    int kernel_size = cbn->transitionalKernels[layer][0].size;
    Kernel * kernels = cbn->transitionalKernels[layer];
    Kernel pooling_kernel  =cbn->poolingKernels[layer];
    BayesianNetwork bn = cbn->bayesianNetworks[layer];

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
            ,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

        freeLayeredImagesContinuos(temp,n_data,d,s);
        d = cbn->bayesianNetworks[l]->depth;
        s = cbn->bayesianNetworks[l]->size;
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

void kernelTrainingWhileUpdatingStructure(ConvolutionalBayesianNetwork cbn, int layer, int iterations, int iterations_update_counts, int iterations_structure_update
        , float learning_rate ,float momentum, int batchSize, float *** images, int * labels, int n_data, int n_incoming_relations , int n_data_used_for_counts, bool verbose){

    if (iterations == 0){
        optimizeStructure(cbn,layer,n_incoming_relations,images,labels,n_data,100,0.03,30,false);
        return;
    }

    if (iterations_update_counts > iterations_structure_update){
        printf("Error: iterations update counts should be smaller than structure update");
        exit(1);
    }
    
    if (iterations_structure_update < iterations){
        cbn->bayesianNetworks[layer]->learning_proportion_white_n_channels = cbn->n_kernels[layer];
        cbn->bayesianNetworks[layer]->learning_proportion_white = malloc(sizeof(float*) * cbn->n_kernels[layer]);
        for (int j = 0; j < cbn->n_kernels[layer]; j++){
            cbn->bayesianNetworks[layer]->learning_proportion_white[j] = malloc(sizeof(float) * 1);
        }

        for (int i  = 0; i < iterations / iterations_structure_update; i++){


            if (verbose) printf("Iteration %d of %d: REvise structure structure\n",i +1, iterations / iterations_structure_update);
            //addRandomStructure(cbn,layer,n_incoming_relations);
            optimizeStructure(cbn,layer,n_incoming_relations,images,labels,n_data,10,0.05,60,false);
            if (verbose) determine_worst_kernel(cbn->bayesianNetworks[layer],true);

            if(verbose) printf("gradient descent for %d iterations\n", iterations_structure_update);
            trainKernelsGradientDescent(cbn,layer,iterations_structure_update,iterations_update_counts,learning_rate,momentum,batchSize,images,labels,n_data,n_data_used_for_counts,verbose);
        }
    }else{
        optimizeStructure(cbn,layer,n_incoming_relations,images,labels,n_data,100,0.03,0,false); 
        trainKernelsGradientDescent(cbn,layer,iterations,iterations_update_counts,learning_rate,momentum,batchSize,images,labels,n_data,n_data_used_for_counts,verbose);
    }
    if (verbose) printf("Done kernel training while updating stucture\n");
}