

#define STOP_IF_CERTAIN_LAYER 0.0 // 0 = no stopping
#define SOOTHE_PREDICTION_NEXT_LAYER 0.6 // 1 = no smoothing

//assumes not stride or padding for now, assumes MUSTIN OUT EITHER KERNEL
//assumes no stride!! (TODO)
// TODO maybe use omp
void propagateChangeInRegionUp(ConvolutionalBayesianNetwork cbn, int lower_x, int lower_y,int upper_x,int upper_y, int layer, int changeId){
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
                    n2->changeID = changeId;
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
        propagateChangeInRegionUp(cbn,min_x_change,min_y_change,max_x_change,max_y_change,layer+1, changeId);
    }
}

//assumes that node n was changed which requires propagating this change in values of higher-layer nodes
void propagatePixelChangeUp(ConvolutionalBayesianNetwork cbn, Node n, int layer, int changeId){
    if (layer == cbn->n_layers){
        //there is not layer above this one
        return;
    }
    propagateChangeInRegionUp(cbn,n->x,n->y,n->x +1, n->y +1,0,changeId);
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

//TODO: Lot's of tuning!
float probabilityPixelTrue(ConvolutionalBayesianNetwork cbn, Node n, bool verbose){
    float prob_current = probabilityGivenParents(n) * probabilityOfChildren(n);
    n->value = ! n->value;
    float prob_other = probabilityGivenParents(n) * probabilityOfChildren(n);
    n->value = ! n->value;
    float prob_true = n->value ? prob_current / (prob_current + prob_other) : prob_other / (prob_current + prob_other);
    if (verbose) printf("probability to be true just based on layer 0 is %f\n", prob_true);
    if (prob_true < STOP_IF_CERTAIN_LAYER || prob_true > 1 - STOP_IF_CERTAIN_LAYER ){
        if (verbose) printf("This is certain enough to not consider the other layers\n");
        return prob_true;
    }

    bool currentValue = n->value;

    int c, counts_nn;
    float * weighted_average_nn_prob_true =  malloc(sizeof(float) * cbn->n_layers);
    float * weighted_average_nn_prob_false =  malloc(sizeof(float) * cbn->n_layers);
    int * counts_for_nn = malloc(sizeof(int) * cbn->n_layers);
    int * nn_in_layers_effected = malloc(sizeof(int) * cbn->n_layers);

    int * n_effected_nodes_layers = malloc(sizeof(int) * cbn->n_layers);
    Node ** effected_nodes_layers = malloc(sizeof(Node *) * cbn->n_layers);
    float ** prob_true_layers = malloc(sizeof(float*)* cbn->n_layers);
    float ** prob_false_layers = malloc(sizeof(float*)* cbn->n_layers);
    int * size_relevant_nodes_layers = malloc(sizeof(int) * cbn->n_layers);
    Node ** relevant_nodes_at_layers = malloc(sizeof(Node *) * cbn->n_layers);
    for(int i = 0; i < cbn->n_layers; i++){
        weighted_average_nn_prob_true[i] = 0;
        weighted_average_nn_prob_false[i]  =0;
        nn_in_layers_effected[i] = 0;
        size_relevant_nodes_layers[i] = 0;
        effected_nodes_layers[i] = effectedNodesOfPixel(cbn,n->x,n->y,i,&n_effected_nodes_layers[i]);
    }


    Node n2, rn;
    NumberNode nn;
    int changeId = rand();
    for (int b = 0; b < 2; b++){
        n->value = !n->value;
        n->changeID = changeId;
        propagatePixelChangeUp(cbn,n,0,changeId);

        for (int i = 0; i < cbn->n_layers; i++){
            counts_for_nn[i] = 0;
            if (b == 0){
                for (int j = 0; j < n_effected_nodes_layers[i]; j++){
                    n2 = effected_nodes_layers[i][j];
                    if (n2->changeID != changeId){
                        //node was not changed in this iteration
                        continue;
                    }
                    size_relevant_nodes_layers[i]++;;
                }
                prob_true_layers[i] = malloc(sizeof(float) * size_relevant_nodes_layers[i]);
                prob_false_layers[i] = malloc(sizeof(float) * size_relevant_nodes_layers[i]);
                relevant_nodes_at_layers[i] = malloc(sizeof(Node) * size_relevant_nodes_layers[i]);
                c = 0;
                for (int j = 0; j < n_effected_nodes_layers[i]; j++){
                    n2 = effected_nodes_layers[i][j];
                    if (n2->changeID != changeId){
                        //node was not changed in this iteration
                        continue;
                    }
                    relevant_nodes_at_layers[i][c] = n2;
                    c++;
                }
            }

            for( int j = 0; j < size_relevant_nodes_layers[i]; j++){

                rn = relevant_nodes_at_layers[i][j];

                if (n->value){
                    prob_true_layers[i][j] = probabilityGivenParents(rn) * probabilityOfChildren(rn);
                }else {
                    prob_false_layers[i][j] = probabilityGivenParents(rn) * probabilityOfChildren(rn);
                }

                for (int k = 0; k < rn->n_numberNodeChildren; k++){
                    nn = rn->numberNodeChildren[k];
                    if ((b == 0 && nn->changeID != changeId) || (b == 1 && nn->changeID == changeId) ){
                        counts_nn = countsOfStateNumberNode(nn);
                        if (n->value){
                            weighted_average_nn_prob_true[i] += counts_nn * probabilityGivenParentsNN(nn);
                        } else {
                            weighted_average_nn_prob_false[i] += counts_nn * probabilityGivenParentsNN(nn);
                        }
                        counts_for_nn[i] += counts_nn;

                        nn->changeID = changeId;

                        if (b == 0){
                            nn_in_layers_effected[i] ++;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < cbn->n_layers; i++){
            if (n->value){
                if (counts_for_nn[i] == 0){
                    weighted_average_nn_prob_true[i] = 0.5;
                }else {
                    if (verbose) printf("True: %f / %d\n",weighted_average_nn_prob_true[i],counts_for_nn[i]);
                    weighted_average_nn_prob_true[i] /= counts_for_nn[i];
                }
            } else {
                if (counts_for_nn[i] == 0){
                    weighted_average_nn_prob_false[i] = 0.5;
                }else {
                    if (verbose) printf("False: %f / %d\n",weighted_average_nn_prob_false[i],counts_for_nn[i]);
                    weighted_average_nn_prob_false[i] /= counts_for_nn[i];
                }
            }
        }
    }

    float average_nn, weighted_average,smoothed_average, overall_result = 0.5, p1,p2;
    for (int i = 0; i < cbn->n_layers; i++){

        if (verbose) {
            printf("In layer %d: \n",i);
            printf("%d nodes could change value, %d did and %d number nodes are effected\n"
                ,n_effected_nodes_layers[i],size_relevant_nodes_layers[i], nn_in_layers_effected[i]); 
        }

        if (size_relevant_nodes_layers[i] == 0){
            if (verbose) printf("break because no change in this or upper layers\n");
            break;
        }

        //results of different nodes are dependent. Use average instead of multiplication
        weighted_average = 0.0;
        for(int j = 0; j <size_relevant_nodes_layers[i]; j++){

            weighted_average += (prob_true_layers[i][j] / (prob_true_layers[i][j] + prob_false_layers[i][j]));
        }

        weighted_average /= size_relevant_nodes_layers[i];
        if (verbose)  printf("Average prob this layer based on pixel stucture: %f \n", weighted_average);

        average_nn = weighted_average_nn_prob_true[i] / (weighted_average_nn_prob_true[i] + weighted_average_nn_prob_false[i]);
        if (verbose && nn_in_layers_effected[i] != 0)  printf("Average prob this layer based on number nodes: %f (true: %f, false: %f)\n", average_nn,weighted_average_nn_prob_true[i], weighted_average_nn_prob_false[i]);

        weighted_average = (weighted_average * average_nn) / (weighted_average * average_nn + (1 - weighted_average) * (1- average_nn));
        if (verbose)  printf("Average prob this layer: %f \n", weighted_average);
        
        if (i != 0){
            //result may be dependent with existing, so make result in this layer less extreme
            p1 = pow(weighted_average, SOOTHE_PREDICTION_NEXT_LAYER );
            p2 = pow(1 - weighted_average, SOOTHE_PREDICTION_NEXT_LAYER );

            smoothed_average = p1 / (p1+p2);
            if (verbose)  printf("Average prob this layer after smoothing: %f \n", smoothed_average);
        }else {
            smoothed_average = weighted_average;
        }

        overall_result = (overall_result * smoothed_average) 
            / (overall_result * smoothed_average + (1-overall_result ) * (1-smoothed_average));

        if (verbose)  printf("Overall result after updating: %f \n", overall_result);

        if (weighted_average < STOP_IF_CERTAIN_LAYER || weighted_average > 1 - STOP_IF_CERTAIN_LAYER){
            //no point in testing next layer
            if (verbose) printf("break because already certain enough to not consider higher layers\n");
            break;
        }
    }

    if (verbose) printf("Final result: %f\n", overall_result);

    for(int i = 0; i < cbn->n_layers; i++){
        free(prob_true_layers[i]);
        free(prob_false_layers[i]);
        free(effected_nodes_layers[i]);
        free(relevant_nodes_at_layers[i]);
    }
    free(n_effected_nodes_layers);
    free(effected_nodes_layers);
    free(prob_true_layers);
    free(prob_false_layers);
    free(size_relevant_nodes_layers);
    free(relevant_nodes_at_layers);
    free(weighted_average_nn_prob_true);
    free(weighted_average_nn_prob_false);
    free(counts_for_nn);
    free(nn_in_layers_effected);

    return overall_result;
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

            probTrue = probabilityPixelTrue(cbn,n,false);

            n->value = (float)rand() / (float)RAND_MAX < probTrue;
            
            if (n->value != value_before){
                propagatePixelChangeUp(cbn,n,0,1);
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

            probTrue = probabilityPixelTrue(cbn,n,false);
            deltaProb = n->value ? - 2 * (probTrue - 0.5 ) : 2 * (probTrue - 0.5 ) ;

            temperature =  iterationToTemperature(iteration,n_iterations);

            if ( deltaProb > 0 || (float)rand() / (float)RAND_MAX < pow(2.71828, deltaProb / temperature)){
                n->value = !n->value;
                propagatePixelChangeUp(cbn,n,0,3);
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

            probTrue = probabilityPixelTrue(cbn,n,false);
            deltaProb = n->value ? - 2 * (probTrue - 0.5 ) : 2 * (probTrue - 0.5 ) ;


            if ( deltaProb > 0){
                n->value = !n->value;
                propagatePixelChangeUp(cbn,n,0,4);
            }
        }

        samples[i] = getImageFromState(cbn);
    }
    return samples;
}


/* void saveBestWorst(bool *** images, int n_images, int n_best ,ConvolutionalBayesianNetwork cbn){
    float * image_probs = malloc(sizeof(float) * n_images);
    for(int i = 0; i < n_images; i++){
        setStateToImage(cbn,images[i]);
        image_probs[i] = logProbabilityStateCBN(cbn); //TODO UPDATE, in order to implement depenent
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
} */