
//assumes not stride or padding for now
//Very inefficient for now!
//TODO: make efficient
void propagatePixelChangeUp(ConvolutionalBayesianNetwork cbn, Node n){
    bool ** image = getImageFromState(cbn);
    setStateToImage(cbn,image);
    freeImage(image,28);
}

double probabilityGivenParents(Node n){
    bool * parent_states = malloc(sizeof(bool) * n->n_parents);
    double result;
    for (int i = 0; i < n->n_parents; i++){
        parent_states[i] = n->parents[i]->value;
    }
    result = n->CPT[binaryToInt(parent_states,n->n_parents)];
    if (! n->value){
        result = 1 - result;
    }
    free(parent_states);
    return result;
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
    double prob_true = 1; // maybe init with very large number to prevent numerical underflow
    double prob_false = 1;
    int n_effected_nodes;

    for (int b = 0; b < 2; b++){
        for (int i = 0; i < cbn->n_layers; i++){
            Node * effectedNodes = effectedNodesOfPixel(cbn,n->x,n->y,0,&n_effected_nodes);
            for (int j = 0; j < n_effected_nodes; j++){
                if (n->value){
                    prob_true *= probabilityGivenParents(n);
                    prob_true *= probabilityOfChildren(n);

                    //printf("True based on own = %f, true based on children = %f\n",probabilityGivenParents(n),probabilityOfChildren(n));
                }else {
                    prob_false *= probabilityGivenParents(n);
                    prob_false *= probabilityOfChildren(n);
                    //rintf("False based on own = %f, False based on children = %f\n",probabilityGivenParents(n),probabilityOfChildren(n));
                }
            }
            free(effectedNodes);
        }
        n->value = !n->value;
        propagatePixelChangeUp(cbn,n); //would maybe be more efficient to only reverse it if new state is really not accepted
    }

    if (prob_false == 0.0 || prob_true == 0.0){
        printf("WARNING: Zero probability, maybe add pseudocounts? Or numerical underflow? ");
    }

    //printf("prob True = %f; prob false = %f\n",prob_true,prob_false);

    return prob_true / (prob_true + prob_false);
}



bool *** gibbsSampling(ConvolutionalBayesianNetwork cbn, int n_samples, int iterations){

    bool *** samples = malloc(sizeof(bool**) * n_samples);
    int image_size = cbn->bayesianNetworks[0]->size;
    int x,y;
    Node n;
    float probTrue;

    samples[0] = getImageFromState(cbn);

    for (int i = 1; i < n_samples; i++){
        for (int j = 0; j < iterations / n_samples; j++){

            x = rand() % image_size;
            y = rand() % image_size;

            n = cbn->bayesianNetworks[0]->nodes[0][x][y];

            probTrue = probabilityPixelTrue(cbn,n);

            if ( (float)rand() / (float)RAND_MAX < probTrue){
                n->value = true;
            }else{
                n->value = false;
            }

            propagatePixelChangeUp(cbn,n);
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
                propagatePixelChangeUp(cbn,n);
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
        for (int j = 0; j < (int)(n_iterations / n_samples); j++){

            x = rand() % image_size;
            y = rand() % image_size;

            n = cbn->bayesianNetworks[0]->nodes[0][x][y];

            probTrue = probabilityPixelTrue(cbn,n);
            deltaProb = n->value ? - 2 * (probTrue - 0.5 ) : 2 * (probTrue - 0.5 ) ;


            if ( deltaProb > 0){
                n->value = !n->value;
                propagatePixelChangeUp(cbn,n);
            }
        }

        samples[i] = getImageFromState(cbn);
    }
    return samples;
}