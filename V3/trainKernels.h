
#define N_THREADS 16
#define NUMBER_NODE_VALUE 10
#define N_TEST_PROPORTION_WHITE 30
#define MAX_CHANGE_WHITE_ITERATION 0.005

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
        return 1.0;
    }


    
    return (float)(n->stateCountsTrue[row_number]) / (float)(counts_row);
}

float prob_child_given_data_dependent_n(Node child, Node m, bool n_value, float *** data){

    Node parent;
    float prob;
    int row_number, counts_row;

    bool own_state =  0.5 < data[child->depth][child->x][child->y];
    bool * parent_states = malloc(sizeof(bool) * child->n_parents);

    for (int j = 0; j < child->n_parents; j++){
        parent = child->parents[j];
        if (parent->x == m->x && parent->y == m->y && parent->depth == m->depth){
            parent_states[j] = n_value;
        } else {
            parent_states[j]  = 0.5 < data[parent->depth][parent->x][parent->y];
        }
    }
    row_number = binaryToInt(parent_states,child->n_parents);
    free(parent_states);

    counts_row = child->stateCountsTrue[row_number] + child->stateCountsFalse[row_number];

    if (counts_row == 0){
        return 1;
    }

    if (own_state){
        return (float)(child->stateCountsTrue[row_number]) / (float)(counts_row);
    }else{
        return (float)(child->stateCountsFalse[row_number]) / (float)(counts_row);
    }
}

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
    //printf("row number %d label %d\n",row_number,number_label);
    counts_row = 1;
    for (int j = 0; j < 10; j++){
        counts_row += nn->stateCounts[row_number][j];
    }

    prob = (float)(nn->stateCounts[row_number][number_label] + 1) / (float)(counts_row);
    
    free(parent_states);
    return prob;
}

//assumes that nn is set to the correct state already
void calculateGradientNode(float *** gradient, float * bias_gradient, Node n, Kernel kernel, Kernel poolingKernel, float *** data_before
    ,int before_depth, int before_size, float *** data_intermediate_before_sigmoid, int size_intermediate, float *** data_after, int number_label){
    
    NumberNode nn;
    int x_pooling, y_pooling, responseX, responseY;
    float prob_if_true, prob_if_false, a,b, value, max_value,s;
    Node child;
    a = 0;

    for (int i = 0; i < n->n_numberNodeChildren; i++){
        nn = n->numberNodeChildren[i];
        prob_if_true = prob_nn_given_data_dependent_n(nn,n,true,number_label,data_after);
        prob_if_false = prob_nn_given_data_dependent_n(nn,n,false,number_label,data_after);

        a += NUMBER_NODE_VALUE * ( prob_if_true  - prob_if_false);
    } 

    
    for(int i = 0; i < n->n_children; i++){
        child = n->children[i];
        prob_if_true = prob_child_given_data_dependent_n(child,n,true,data_after);
        prob_if_false = prob_child_given_data_dependent_n(child,n,false,data_after);
        a += prob_if_true  - prob_if_false;
    } 
     

    //prob_if_true = prob_n_given_data(n,data_after);
    a += prob_if_true - (1 - prob_if_true); 

    a /= (1 + n->n_children + n->n_numberNodeChildren * NUMBER_NODE_VALUE);

    b = average_prob_true_n(n);
    b = (b - (1 - b));
    a = (a - b); 

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
    s = sigmoid(data_intermediate_before_sigmoid[n->depth][x_pooling][y_pooling]);

    //printf("value of node is %f, and after sigmoid %f (%f)\n",data_intermediate_before_sigmoid[n->depth][x_pooling][y_pooling],s, data_after[n->depth][n->x][n->y] );
    
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
                full_transformation[i][x][y] = sigmoid(full_transformation[i][x][y]);
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

    
    float extreme_value = 0.0;

    //normalize gradient
    for (int i = 0; i < n_kernels; i++){
        if ( extreme_value < fabs(bias_gradient[i])){
            extreme_value = fabs(bias_gradient[i]);
        }
        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){
                    if ( extreme_value < fabs(gradient[i][d][x][y])){
                        extreme_value = fabs(gradient[i][d][x][y]);
                    }
                }
            }
        }
    }
    for (int i = 0; i < n_kernels; i++){
        bias_gradient[i] *=  learning_rate / extreme_value;
        for (int d = 0; d < kernel_depth; d++){
            for (int x  =0; x < kernel_size; x++){
                for (int y  =0; y < kernel_size; y++){
                    gradient[i][d][x][y] *= learning_rate /extreme_value ;
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
            data_after = applyConvolutionWeighted(data_previous_layer[j],s,s,kernels[i],true);
            for (int x = 0;  x < size_after; x++){
                for (int y = 0; y < size_after; y++){
                    proportion_white_kernels[i] += 0.5 < data_after[x][y] ? 1.0 : 0.0;
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
    float change = 1.0;
    bool update;
    for (int iteration = 0; iteration < 20; iteration++){
        update = false;

        if (iteration != 0){
            update_proportion_white(new_proportion_white,kernels,n_kernels,data_previous_layer, n_data, d, s);
        }

        for (int i = 0; i < n_kernels; i++){
            
            if (should_increase[i]){
                //should increase...
                if (new_proportion_white[i] < proportion_white_kernels[i]){
                    //... but does not. Increase bias!
                    kernels[i].bias += change;
                    update = true;
                } else if ( proportion_white_kernels[i] + MAX_CHANGE_WHITE_ITERATION < new_proportion_white[i]){
                    //.. but not that much. Decrease bias!
                    kernels[i].bias -= change;
                    update = true;
                } else {
                    //.. and does not too much. No change required
                }
            }else{
                //should decrease...
                if (proportion_white_kernels[i] < new_proportion_white[i]){
                    // ... but does not. Decrease bias
                    kernels[i].bias -= change;
                    update = true;
                } else if (  new_proportion_white[i] + MAX_CHANGE_WHITE_ITERATION < proportion_white_kernels[i]){
                    //... but not that much. Increase bias!
                    kernels[i].bias += change;
                    update = true;
                } else {
                    //.. and does not too much. No change required
                }
            }
        }
        if (! update){
            printf("stop after iteration %d, (after change = %f)\n", iteration, change * 2);
            break;
        }

        change /=2;
    }

    free(new_proportion_white);
    free(should_increase);
}


void trainKernelsGradientDescent(ConvolutionalBayesianNetwork cbn, int layer, int iterations, float learning_rate ,float momentum, int batchSize
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

    float * proportion_white_kernels = malloc(sizeof(float) * n_kernels);

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

    for (int it = 0; it< iterations; it++){

        if (verbose) printf("GRAD_DEC: iteration %d of %d\n",it,iterations);

        if (verbose){
            int n_test_data = 500 < n_data ? 500 : n_data;
            float **** test_data = dataTransition(data_previous_layer,n_test_data,d,s,kernels,n_kernels,pooling_kernel);

            fitDataCounts(bn,test_data,n_test_data); 
            for (int i = 0; i < bn->n_numberNodes; i++){
                fitDataCountsNumberNode(bn->numberNodes[i],test_data,labels, n_test_data);
            }

            /*  float logLNumberNodes = 0;
            for (int i = 0; i < bn->n_numberNodes; i++){
                logLNumberNodes += logMaxLikelihoodDataNumberNode(bn->numberNodes[i]);
            }

            float logLNumberNodesNoRelations = 0;
            for (int i = 0; i < bn->n_numberNodes; i++){
                logLNumberNodesNoRelations += logLikelihoodDataNumberNodeNoRelations(bn->numberNodes[i]);
            }

            printf("log probability nn test data with relations = %.0f (%.0f with no relations)\n", logLNumberNodes, logLNumberNodesNoRelations);

            float log_prob_data_relations = logMaxLikelihoodDataGivenModel(bn,NULL,0, false);
            float log_prob_data_no_relations = logLikelihoodDataGivenModelNoRelations(bn);

            printf("log probability n with - without relations %f \n", log_prob_data_relations - log_prob_data_no_relations);  */

            float average_prob_with_relations = average_probability_nodes(bn);
            float average_prob_no_relations = average_probability_nodes_no_relations(bn);

            printf("Average prob no relations %f, average prob with relations %f difference: %f\n", average_prob_no_relations, average_prob_with_relations
                    , average_prob_with_relations - average_prob_no_relations);

            if (bn->learning_curve_len == bn->learning_curve_size){
                bn->learning_curve_len *=2;
                bn->learning_curve = realloc(bn->learning_curve, sizeof(float) * bn->learning_curve_len);
                bn->learning_proportion_white = realloc(bn->learning_proportion_white, sizeof(float) * bn->learning_curve_len);
            }
            bn->learning_curve[bn->learning_curve_size] = average_prob_with_relations - average_prob_no_relations;
            
            bn->learning_proportion_white[bn->learning_curve_size] = proportionWhite(test_data,n_test_data,n_kernels,bn->size);
            bn->learning_curve_size +=1;

            freeLayeredImagesContinuos(test_data,n_test_data,n_kernels,bn->size);
        }

        if (verbose) printf("GRAD_DEC: transform subset data (subset size = %d)\n", n_data_used_for_counts);
        dataTransitionSubset(data_previous_layer,n_data,d,s,kernels,n_kernels,pooling_kernel,n_data_used_for_counts,subset_data,labels, subset_labels);

        if (verbose) printf("GRAD_DEC: fit data counts!n\n");
        fitDataCounts(bn,subset_data,n_data_used_for_counts); 
        for (int j = 0; j < bn->n_numberNodes; j++){
            fitDataCountsNumberNode(bn->numberNodes[j],subset_data,subset_labels, n_data_used_for_counts);
        }
        

        if (verbose) printf("GRAD_DEC: free subset\n");
        for (int j = 0; j < n_data_used_for_counts; j++){
            freeImagesContinuos(subset_data[j],bn->depth,bn->size);
        }

        if (verbose) printf("GRAD_DEC: learn gradien\n");
        calculateGradient(bn, gradient,bias_gradient,n_kernels,data_previous_layer,n_data,s, labels, kernels,pooling_kernel,learning_rate,batchSize);

        if (verbose) printf("GRAD_DEC: combine running gradient\n");
        additionGradients(previousGradient,gradient,momentum,n_kernels,kernel_depth,kernel_size);
        for (int j = 0; j < n_kernels; j++){
            previous_bias_gradient[j] = momentum * previous_bias_gradient[j] + (1 - momentum) * bias_gradient[j];
        }

        update_proportion_white(proportion_white_kernels, kernels, n_kernels, data_previous_layer, n_data, d , s );

        if (verbose) printf("GRAD_DEC: update kernels\n");
        updateKernels(kernels,n_kernels,kernel_depth,kernel_size,previousGradient, previous_bias_gradient);

        //adapts biases if change in white too large
        moderateBiases(proportion_white_kernels,kernels, n_kernels, data_previous_layer, n_data, d , s );
    }

    if (verbose) printf("GRAD_DEC: done -> free everything\n");
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

void kernelTrainingWhileUpdatingStructure(ConvolutionalBayesianNetwork cbn, int layer, int iterations, int iterations_structure_update, float learning_rate 
        ,float momentum, int batchSize, float *** images, int * labels, int n_data, int n_incoming_relations , int n_data_used_for_counts, bool verbose){

    if (iterations == 0){
        optimizeStructure(cbn,layer,n_incoming_relations,images,labels,n_data,100,0.03,30,false);
        return;
    }
    
    if (iterations_structure_update < iterations){
        for (int i  = 0; i < iterations / iterations_structure_update; i++){

            if (verbose) printf("Iteration %d of %d: REvise structure structure\n",i +1, iterations / iterations_structure_update);
            //addRandomStructure(cbn,layer,4);
            optimizeStructure(cbn,layer,n_incoming_relations,images,labels,n_data,100,0.01,30,false);

            if(verbose) printf("gradient descent for %d iterations\n", iterations_structure_update);
            trainKernelsGradientDescent(cbn,layer,iterations_structure_update,learning_rate,momentum,batchSize,images,labels,n_data,n_data_used_for_counts,true);
        }
    }else{
        optimizeStructure(cbn,layer,n_incoming_relations,images,labels,n_data,100,0.03,0,false); 
        trainKernelsGradientDescent(cbn,layer,iterations,learning_rate,momentum,batchSize,images,labels,n_data,n_data_used_for_counts,true);
    }
    if (verbose) printf("Done kernel training while updating stucture\n");
}