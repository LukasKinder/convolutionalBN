

//assumes not stride or padding for now, assumes MUSTIN OUT EITHER KERNEL
//assumes no stride!! (TODO)
// TODO maybe use omp
void propagateChangeInRegionUp(ConvolutionalBayesianNetwork cbn, int lower_x, int lower_y,int upper_x,int upper_y, int layer){
    if (layer >= cbn->n_layers-1){
        //there is not layer above this one
        return;
    }

    int min_x_change = 1000;
    int min_y_change = 1000;
    int max_x_change = -1;
    int max_y_change = -1;
    bool value_changed = false;

    int pooling_kernel_size = cbn->poolingKernels[layer].size;
    int transition_kernel_size = cbn->transitionalKernels[layer][0].size;

    int upper_left_corner_image_x = lower_x - (transition_kernel_size -1) - (pooling_kernel_size -1);
    int upper_left_corner_image_y = lower_y - (transition_kernel_size -1) - (pooling_kernel_size -1);

    int size_after_transitional_x = (upper_x - lower_x) -1 + transition_kernel_size +  2 * (pooling_kernel_size -1);
    int size_after_transitional_y = (upper_y - lower_y) -1 + transition_kernel_size +  2 * (pooling_kernel_size -1);

    int size_after_pooling_x = size_after_transitional_x -( pooling_kernel_size) +1;
    int size_after_pooling_y = size_after_transitional_y -( pooling_kernel_size) +1;

    bool ** after_transitional_representation = malloc(sizeof(bool*) * size_after_transitional_x);
    for(int i = 0; i < size_after_transitional_x; i++){
        after_transitional_representation[i] = malloc(sizeof(bool*) * size_after_transitional_y);
    }

    bool is_true;
    Kernel k;
    Node n2;
    BayesianNetwork bn_lower = cbn->bayesianNetworks[layer];
    for (int d = 0; d < cbn->n_kernels[layer]; d++){
        k = cbn->transitionalKernels[layer][d];

        for(int x =0; x < size_after_transitional_x; x++){
            if (upper_left_corner_image_x + x  <0  || upper_left_corner_image_x + x + k.size -1 >=bn_lower->size) continue;

            for(int y = 0; y < size_after_transitional_y; y++){
                if (upper_left_corner_image_y + y  <0  || upper_left_corner_image_y + y + k.size -1 >=bn_lower->size) continue;

                is_true = true;


                for(int d_kernel = 0; d_kernel < k.depth; d_kernel++){
                    for(int x_kernel = 0; x_kernel < k.size; x_kernel++){
                        for(int y_kernel = 0; y_kernel < k.size; y_kernel++){

                            n2 = cbn->bayesianNetworks[layer]->nodes[d_kernel][upper_left_corner_image_x + x + x_kernel][upper_left_corner_image_y + y + y_kernel];
                            if (n2->value && k.map[d_kernel][x_kernel][y_kernel] == must_false){
                                is_true = false;
                            }
                            if (!n2->value && k.map[d_kernel][x_kernel][y_kernel] == must_true){
                                is_true = false;
                            }
                        }
                    }
                }
                after_transitional_representation[x][y] = is_true;
            }
        }
        k = cbn->poolingKernels[layer];


        for(int x =0; x < size_after_pooling_x; x++){
            if (upper_left_corner_image_x + x < 0 || upper_left_corner_image_x + x >= cbn->bayesianNetworks[layer+1]->size) continue;
            for(int y = 0; y < size_after_pooling_y; y++){
                if (upper_left_corner_image_y + y < 0 || upper_left_corner_image_y + y >= cbn->bayesianNetworks[layer+1]->size) continue;

                is_true = false;
                for(int x_kernel = 0; x_kernel < k.size; x_kernel++){
                    for(int y_kernel = 0; y_kernel < k.size; y_kernel++){
                        is_true =  is_true || after_transitional_representation[x + x_kernel ][y + y_kernel];
                    }
                }
                //node  on layer above that should hode this value
                n2 = cbn->bayesianNetworks[layer +1]->nodes[d][upper_left_corner_image_x + x][upper_left_corner_image_y + y];
                if (n2->value != is_true){
                    n2->value = is_true;
                    value_changed = true;
                    if (min_x_change > n2->x) min_x_change = n2->x;
                    if (min_y_change > n2->y) min_y_change = n2->y;
                    if (max_x_change < n2->x) max_x_change = n2->x;
                    if (max_y_change < n2->y) max_y_change = n2->y;
                }
            }
        }
    }

    for(int i = 0; i < size_after_transitional_x; i++){
        free(after_transitional_representation[i]);
    }
    free(after_transitional_representation);

    if (value_changed){
        propagateChangeInRegionUp(cbn,min_x_change,min_y_change,max_x_change,max_y_change,layer+1);
    }
}

//assumes that node n was changed which requires propagating this change in values of higher-layer nodes
void propagatePixelChangeUp(ConvolutionalBayesianNetwork cbn, Node n, int layer){
    if (layer == cbn->n_layers){
        //there is not layer above this one
        return;
    }
    propagateChangeInRegionUp(cbn,n->x,n->y,n->x +1, n->y +1,0);
}


//*migh suffer from numerical underflow
double probabilityOfChildren(Node n){
    double result = 1;
    for (int i = 0; i < n->n_children; i++){
        result *= probabilityGivenParents(n->children[i]);
    }
    return result;
}

//assumes not stride or padding for now
//returns array of Nodes
Node * effectedNodesOfPixel(ConvolutionalBayesianNetwork cbn, int pixel_x, int pixel_y, int layer, int * n_effected_nodes){
    int effectFieldSize = 1;

    int size_layer = cbn->bayesianNetworks[layer]->size;
    int depth_layer = cbn->bayesianNetworks[layer]->depth;

    for (int i = 0; i < layer; i++){
        effectFieldSize += cbn->transitionalKernels[i][0].size -1;
        effectFieldSize += cbn->poolingKernels[i].size -1;
    }

    *n_effected_nodes = 0;
    Node * effectedNodes = malloc(sizeof(Node) *effectFieldSize * effectFieldSize * depth_layer);

    for (int x = pixel_x; x > pixel_x - effectFieldSize; x--){
        for (int y = pixel_y; y > pixel_y - effectFieldSize; y--){

            if (x < 0 || x >= size_layer || y < 0 || y >=  size_layer ){
                //node does not exist
                continue;
            }

            for (int d = 0 ; d < depth_layer; d++){
                effectedNodes[*n_effected_nodes] = cbn->bayesianNetworks[layer]->nodes[d][x][y];
                (*n_effected_nodes)++;
            }
        }
    }
    return effectedNodes;
}

//might suffer from numerical underflow
float probabilityPixelTrue(ConvolutionalBayesianNetwork cbn, Node n){
    bool currentValue = n->value;
    int n_effected_nodes;

    float * prob_true_layers = calloc(sizeof(float), cbn->n_layers);
    float* prob_false_layers = calloc(sizeof(float), cbn->n_layers);

    Node n2;
    double intermediate;

    int n_effected_nodes_layer[10]; //Todo remove

    for (int b = 0; b < 2; b++){
        for (int i = 0; i < cbn->n_layers; i++){
            Node * effectedNodes = effectedNodesOfPixel(cbn,n->x,n->y,i,&n_effected_nodes);
            n_effected_nodes_layer[i] = n_effected_nodes;
            for (int j = 0; j < n_effected_nodes; j++){
                n2 = effectedNodes[j];
                if (n->value){

                    intermediate = probabilityGivenParents(n2);
                    prob_true_layers[i] += intermediate;
                    intermediate = probabilityOfChildren(n2);
                    prob_true_layers[i] += intermediate;

                    //printf("True based on own = %f, true based on children = %f\n",probabilityGivenParents(n),probabilityOfChildren(n));
                }else {
                    
                    intermediate = probabilityGivenParents(n2);
                    prob_false_layers[i] += intermediate;
                    intermediate = probabilityOfChildren(n2);
                    prob_false_layers[i] += intermediate;
                    //rintf("False based on own = %f, False based on children = %f\n",probabilityGivenParents(n),probabilityOfChildren(n));
                }
            }
            free(effectedNodes);
        }
        n->value = !n->value;
        propagatePixelChangeUp(cbn,n,0); //would maybe be more efficient to only reverse it if new state is really not accepted
    }

    SuperSmall prob_true = initSuperSmall(1.0); // maybe init with very large number to prevent numerical underflow
    SuperSmall prob_false = initSuperSmall(1.0);
    for (int i = 0; i < cbn->n_layers; i++){
        //additive or multiplicative???

        prob_true_layers[i] /= n_effected_nodes_layer[i] *2;
        prob_false_layers[i] /= n_effected_nodes_layer[i] *2;

        multiplySuperSmallF(&prob_true, prob_true_layers[i]);
        multiplySuperSmallF(&prob_false, prob_false_layers[i]);
        //prob_true = addSupersmalls(prob_true, prob_true_layers[i]);
        //prob_false = addSupersmalls(prob_false, prob_false_layers[i]);

        printf("Based on layer %d, prob true = %f (true:%f / false:%f) ",i
            ,prob_true_layers[i] / (prob_true_layers[i] +prob_false_layers[i]) 
            ,prob_true_layers[i], prob_false_layers[i]); 
        printf("n_ effected nodes = %d\n",n_effected_nodes_layer[i]);

    }

    //printf("prob True = %f; prob false = %f\n",prob_true,prob_false);

    return divideSuperSmalls(prob_true,  addSupersmalls(prob_true, prob_false) ) ;
}



bool *** gibbsSampling(ConvolutionalBayesianNetwork cbn, int n_samples, int iterations){

    bool *** samples = malloc(sizeof(bool**) * n_samples);
    int image_size = cbn->bayesianNetworks[0]->size;
    int x,y;
    Node n;
    float probTrue;
    bool value_before;

    samples[0] = getImageFromState(cbn);

    for (int i = 1; i < n_samples; i++){
        printf("%d%\n",i);
        for (int j = 0; j < iterations / n_samples; j++){

            x = rand() % image_size;
            y = rand() % image_size;

            n = cbn->bayesianNetworks[0]->nodes[0][x][y];
            value_before = n->value;

            probTrue = probabilityPixelTrue(cbn,n);

            n->value = (float)rand() / (float)RAND_MAX < probTrue;
            
            if (n->value != value_before){
                propagatePixelChangeUp(cbn,n,0);
            }
        }

        samples[i] = getImageFromState(cbn);
    }
    return samples;
}

//exponentially decreasing temperature, reached 0.005 at max iterations
float iterationToTemperature(float iteration, float maxIterations){
    return pow( pow(2.71828, -5.298317 / maxIterations) ,iteration);
}

bool *** simulatedAnnealing(ConvolutionalBayesianNetwork cbn, int n_samples, int n_iterations){

    bool *** samples = malloc(sizeof(bool**) * n_samples);
    int image_size = cbn->bayesianNetworks[0]->size;
    int x,y;
    Node n;
    float probTrue, deltaProb, temperature;
    int iteration = 0;

    samples[0] = getImageFromState(cbn);

    for (int i = 1; i < n_samples; i++){
        printf("%d%\n",i);
        for (int j = 0; j < (int)(n_iterations / n_samples); j++){
            iteration++;

            x = rand() % image_size;
            y = rand() % image_size;

            n = cbn->bayesianNetworks[0]->nodes[0][x][y];

            probTrue = probabilityPixelTrue(cbn,n);
            deltaProb = n->value ? - 2 * (probTrue - 0.5 ) : 2 * (probTrue - 0.5 ) ;

            temperature =  iterationToTemperature(iteration,n_iterations);

            if ( deltaProb > 0 || (float)rand() / (float)RAND_MAX < pow(2.71828, deltaProb / temperature)){
                n->value = !n->value;
                propagatePixelChangeUp(cbn,n,0);
            }
        }

        samples[i] = getImageFromState(cbn);
    }
    return samples;
}

bool *** strictClimbing(ConvolutionalBayesianNetwork cbn, int n_samples, int n_iterations){

    bool *** samples = malloc(sizeof(bool**) * n_samples);
    int image_size = cbn->bayesianNetworks[0]->size;
    int x,y;
    Node n;
    float probTrue, deltaProb;

    samples[0] = getImageFromState(cbn);

    for (int i = 1; i < n_samples; i++){
        printf("%d%\n",i);
        for (int j = 0; j < (int)(n_iterations / n_samples); j++){

            x = rand() % image_size;
            y = rand() % image_size;

            n = cbn->bayesianNetworks[0]->nodes[0][x][y];

            probTrue = probabilityPixelTrue(cbn,n);
            deltaProb = n->value ? - 2 * (probTrue - 0.5 ) : 2 * (probTrue - 0.5 ) ;


            if ( deltaProb > 0){
                n->value = !n->value;
                propagatePixelChangeUp(cbn,n,0);
            }
        }

        samples[i] = getImageFromState(cbn);
    }
    return samples;
}


void saveBestWorst(bool *** images, int n_images, int n_best ,ConvolutionalBayesianNetwork cbn){
    float * image_probs = malloc(sizeof(float) * n_images);
    for(int i = 0; i < n_images; i++){
        setStateToImage(cbn,images[i]);
        image_probs[i] = logProbabilityStateCBN(cbn);
    }

    //save the best,worst

    char name_best[10] = "bestX";
    char name_worst[10] = "worstX";
    int best_index,worst_index;
    float current_best,current_worst, last_best = 0,last_worst = -999999999;
    for (int i  =0; i < n_best; i++ ){
        current_best = -999999999;
        current_worst = 0;
        for( int j = 0; j < n_images; j++){
            if (image_probs[j] > current_best && image_probs[j] < last_best){
                current_best = image_probs[j];
                best_index = j;
            }
            if (image_probs[j] < current_worst && image_probs[j] > last_worst){
                current_worst = image_probs[j];
                worst_index = j;
            }
        }
        last_best = current_best;
        last_worst = current_worst;


        name_best[4] = '0' + i;
        saveImage(images[best_index],28,name_best);

        name_worst[5] = '0' + i;
        saveImage(images[worst_index],28,name_worst);
        

    }
    free(image_probs);
}