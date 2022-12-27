
#define NUMBER_NODE_DATA_PER_CONFIGURATION 100
#define NUMBER_NODE_PARENT_SUBSET_SIZE 0.01

void setValuesNumberNode(ConvolutionalBayesianNetwork cbn, int value){
    BayesianNetwork bn;
    for (int i = 0; i < cbn->n_layers; i++){
        bn = cbn->bayesianNetworks[i];
        for (int j = 0; j < bn->n_numberNodes; j++){
            bn->numberNodes[j]->value = value;
        }
    }
}


void sampleData(bool **** data, int * data_labels ,int n_data, bool **** sample, int * sample_labels, int n_samples){
    int rand_index;
    for (int i = 0; i < n_samples; i++){
        rand_index = rand() % n_data;
        sample[i] = data[rand_index];
        sample_labels[i] = data_labels[rand_index];
    }
}


//requires that the state counts are already known
double logMaxLikelihoodDataNumberNode(NumberNode nn){

    int n_rows = pow(2, nn->n_parents);
    int n_counts_row;
    double result = 0;
    for (int i = 0; i < n_rows; i++){
        n_counts_row = 0;
        for(int j = 0; j < 10; j++){
            n_counts_row += nn->stateCounts[i][j];
        }

        for(int j = 0; j< 10;j++){
            if (nn->stateCounts[i][j] != 0){
                result += (float)(nn->stateCounts[i][j]) * log( (float)(nn->stateCounts[i][j]) / (float)(n_counts_row));
            }
        }
    }
    return result;
}

void learnStructureNumberNodes(int n_nodes, int n_relations, int layer, ConvolutionalBayesianNetwork cbn, bool *** images, int * number_labels, int n_data, bool verbose){

    //checking for potential problems:
    BayesianNetwork bn = cbn->bayesianNetworks[layer];
    if (bn->numberNodes != NULL){
        printf("Error: number nodes already exist");
        exit(0);
    }
    int data_needed =  NUMBER_NODE_DATA_PER_CONFIGURATION * pow(2, n_relations -1);
    if (data_needed > n_data){
        printf("Warning: not enough data for number node structure search with %d relations\n", n_relations);
    }

    if (verbose) printf("LEARN_NN: inti: \n");

    //init number nodes
    NumberNode nn;
    int n_parent_combinations = pow(2, n_relations);
    bn->numberNodes = malloc(sizeof(NumberNode) * n_nodes);
    bn->n_numberNodes = n_nodes;
    for (int i = 0; i < n_nodes; i++){
        nn = initNumberNode();
        bn->numberNodes[i] = nn;

        nn->parents = malloc(sizeof(Node) * n_relations);
        nn->stateCounts = malloc(sizeof(int *) * n_parent_combinations);
        for (int j = 0; j < n_parent_combinations; j++){
            nn->stateCounts[j] = malloc(sizeof(int) * 10); //for 0-9
        }
    }

    if (verbose) printf("LEARN_NN: initialization done \n Transform data to right layer:\n");


    //set data to data of layer
    bool ****temp, **** layered_images = imagesToLayeredImages(images,n_data,28);
    int d,s;
    for (int l = 0; l < layer; l++){
        temp = layered_images;
        d = cbn->bayesianNetworks[l]->depth;
        s = cbn->bayesianNetworks[l]->size;
        layered_images = dataTransition(temp,n_data,d,s
            ,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

        freeLayeredImages(temp,n_data,d,s);
    }


    if (verbose) printf("LEARN_NN: finished data transformation\n \n");


    //#pragma omp parallel for private(nn)
    for (int i = 0; i < n_nodes; i++){

        if (verbose) printf("\tLEARN_NN: learn NN %d of %d \n",i,n_nodes);

        bool ****bootstrapData = malloc(sizeof(bool ***) * data_needed);
        int * bootstrapNumberLabels = malloc(sizeof(int ) * data_needed);
        int bootstrapSize;
        Node n, best_parent;
        float current, best_heuristic;


        nn = bn->numberNodes[i];
        sampleData(layered_images, number_labels ,n_data, bootstrapData,bootstrapNumberLabels ,data_needed);

        if (verbose) printf("\tLEARN_NN: init bootstrap done  \n");
        for (int r = 0; r < n_relations; r++){
            if (verbose) printf("\t\tLEARN_NN: adding relation %d of %d \n",r,n_relations);

            bootstrapSize = NUMBER_NODE_DATA_PER_CONFIGURATION * pow(2, r);

            best_heuristic = -999999999999;
            for (int d = 0; d < bn->depth; d++){
                for (int x  =0; x < bn->size; x++){
                    for (int y  = 0; y < bn->size; y++){

                        if ((float)rand() / (float)(RAND_MAX ) > NUMBER_NODE_PARENT_SUBSET_SIZE) continue;

                        if (verbose) printf("\t\t\tLEARN_NN: mode ad %d %d %d considered \n",d,x,y);

                        n = bn->nodes[d][x][y]; 
                        nn->parents[nn->n_parents] = n;
                        (nn->n_parents)++;

                        //add counts
                        if (verbose) printf("\t\t\tLEARN_NN: add coutns\n");
                        fitDataCountsNumberNode(nn,bootstrapData,bootstrapNumberLabels,bootstrapSize);
                        //measure goodness
                        if (verbose) printf("\t\t\tLEARN_NN: calculate heuristic\n");
                        current = logMaxLikelihoodDataNumberNode(nn);
                        (nn->n_parents)--;

                        if (current > best_heuristic){
                            best_heuristic = current;
                            best_parent = n;
                            if (verbose) printf("\t\t\tLEARN_NN: new best!\n");
                        }



                    }
                }
            }
            if (verbose) printf("\t\tLEARN_NN: add relation\n");
            nn->parents[nn->n_parents] = best_parent;
            (nn->n_parents)++;

            best_parent->numberNodeChildren = realloc(best_parent->numberNodeChildren, sizeof(NumberNode) *  (best_parent->n_numberNodeChildren + 1));
            best_parent->numberNodeChildren[best_parent->n_numberNodeChildren] = nn;
            best_parent->n_numberNodeChildren++;
        }
        if (verbose) printf("\tLEARN_NN: learn NN %d complete\n",i);
        free(bootstrapData);
        free(bootstrapNumberLabels);
    }
    freeLayeredImages(layered_images,n_data,bn->depth,bn->size);
}
