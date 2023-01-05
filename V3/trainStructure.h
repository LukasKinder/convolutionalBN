
#define NUMBER_NODE_DATA_PER_CONFIGURATION 200
#define STRUCTURE_SEARCH_DATA_PER_CONFIGURATION 40
#define NUMBER_NODE_PARENT_SUBSET_SIZE 0.03

void setValuesNumberNode(ConvolutionalBayesianNetwork cbn, int value){
    BayesianNetwork bn;
    for (int i = 0; i < cbn->n_layers; i++){
        bn = cbn->bayesianNetworks[i];
        for (int j = 0; j < bn->n_numberNodes; j++){
            bn->numberNodes[j]->value = value;
        }
    }
}



void sampleData(float **** data, int * data_labels ,int n_data, float **** sample, int * sample_labels, int n_samples){
    int rand_index;
    for (int i = 0; i < n_samples; i++){
        rand_index = rand() % n_data;
        sample[i] = data[rand_index];
        sample_labels[i] = data_labels[rand_index];
    }
}

void learnStructureNumberNodes(int n_relations, int layer, BayesianNetwork bn, float **** layered_images, int * number_labels, int n_data, bool verbose){

    //checking for potential problems:
    int n_nodes  =bn->n_numberNodes;
    if (bn->numberNodes == NULL){
        printf("Error: number nodes do not exist");
        exit(0);
    }
    int data_needed =  NUMBER_NODE_DATA_PER_CONFIGURATION * pow(2, n_relations -1);
    if (data_needed > n_data){
        printf("Warning: not enough data for number node structure search with %d relations\n", n_relations);
        data_needed = n_data;
    }

    if (verbose) printf("LEARN_NN: inti: \n");

    //init number nodes and reset Structure
    NumberNode nn;
    int n_parent_combinations = pow(2, n_relations);
    for (int i = 0; i < n_nodes; i++){
        nn = bn->numberNodes[i];
        nn->n_parents = 0;

        if (nn->parents == NULL) nn->parents = malloc(sizeof(Node) * n_relations);
        if (nn->stateCounts == NULL){
            nn->stateCounts = malloc(sizeof(int *) * n_parent_combinations);
            for (int j = 0; j < n_parent_combinations; j++){
                nn->stateCounts[j] = malloc(sizeof(int) * 10); //for 0-9
            }
        } 
    }
    for (int d = 0; d < bn->depth; d++){
        for (int x = 0; x < bn->size; x++){
            for (int y = 0; y < bn->size; y++){
                bn->nodes[d][x][y]->n_numberNodeChildren = 0;
            }
        }
    }



    if (verbose) printf("LEARN_NN: finished data transformation\n \n");


    #pragma omp parallel for private(nn)
    for (int i = 0; i < n_nodes; i++){

        if (verbose) printf("\tLEARN_NN: learn NN %d of %d \n",i,n_nodes);

        float ****bootstrapData = malloc(sizeof(float ***) * data_needed);
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
            bootstrapSize = bootstrapSize > n_data ? n_data : bootstrapSize;

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
}

void addRelationChildParent(BayesianNetwork bn, int d_child, int position_parent, bool add_child_relation){
    Node child, parent;
    int distance_vertically, distance_horizontally;
    int d_parent = position_parent % bn->depth;

    if (! bn->diagonals){
        if (position_parent / bn->depth == 0){
            //up relation
            distance_vertically =  - bn->distanceRelation;
            distance_horizontally = 0;
        } else{
            //left relation
            distance_vertically = 0;
            distance_horizontally = - bn->distanceRelation;
        }
    } else {
        if (position_parent / bn->depth == 0){
            //up-felt relation
            distance_vertically = - bn->distanceRelation;
            distance_horizontally = - bn->distanceRelation;
        } else if (position_parent / bn->depth == 1){
            distance_vertically =  - bn->distanceRelation;
            distance_horizontally = 0;
        } else if (position_parent / bn->depth == 2){
            //up-rigth relation
            distance_vertically = -bn->distanceRelation;
            distance_horizontally = bn->distanceRelation;
        } else{
            //left relation
            distance_vertically = 0;
            distance_horizontally = - bn->distanceRelation;
        }
    }

    
    for (int x = 0; x < bn->size; x++){
        for (int y = 0; y < bn->size; y++){

            if (x + distance_vertically < 0 || y + distance_horizontally < 0
                || x + distance_vertically >= bn->size || y + distance_horizontally >= bn->size) continue; //can not add relations

            child = bn->nodes[d_child][x][y];
            parent = bn->nodes[d_parent][x + distance_vertically][y + distance_horizontally];

            if (child->parents == NULL){
                printf("Error: parent array of node is NULL\n");
                exit(1);
            }

            child->parents[child->n_parents] = parent;
            (child->n_parents)++;

            if (add_child_relation){
                parent->children[parent->n_children] = child;
                (parent->n_children)++;
            }
        }
    }
}

//reverses change by "addRelationChildParent"
void removeLastRelationChildParent(BayesianNetwork bn, int d_child, int position_parent, bool remove_child_relation){
    Node child, parent;
    int distance_vertically, distance_horizontally;
    int d_parent = position_parent % bn->depth;

    if (! bn->diagonals){
        if (position_parent / bn->depth == 0){
            //up relation
            distance_vertically =  - bn->distanceRelation;
            distance_horizontally = 0;
        } else{
            //left relation
            distance_vertically = 0;
            distance_horizontally = - bn->distanceRelation;
        }
    } else {
        if (position_parent / bn->depth == 0){
            //up-felt relation
            distance_vertically = - bn->distanceRelation;
            distance_horizontally = - bn->distanceRelation;
        } else if (position_parent / bn->depth == 1){
            distance_vertically =  - bn->distanceRelation;
            distance_horizontally = 0;
        } else if (position_parent / bn->depth == 2){
            //up-rigth relation
            distance_vertically = -bn->distanceRelation;
            distance_horizontally = bn->distanceRelation;
        } else{
            //left relation
            distance_vertically = 0;
            distance_horizontally = - bn->distanceRelation;
        }
    }
    
    for (int x = 0; x < bn->size; x++){
        for (int y = 0; y < bn->size; y++){

            if (x + distance_vertically < 0 || y + distance_horizontally < 0
                || x + distance_vertically >= bn->size || y + distance_horizontally >= bn->size) continue; 

            child = bn->nodes[d_child][x][y];
            (child->n_parents)--;

            if (remove_child_relation){
                parent = bn->nodes[d_parent][x + distance_vertically][y + distance_horizontally];
                (parent->n_children)--;
            }
        }
    }
}


bool optimizeStructureUsingStructureHeuristic(BayesianNetwork bn,  float **** data, int n_data
        , float *** heuristics, int n_incoming_relations, bool verbose){

    int depth = bn->depth;
    int n_possibly_incoming_directions = bn->diagonals ? 4 * bn->depth : 2 * bn->depth;

    //remove all relations
    Node n;
    for (int i = 0; i < bn->depth; i++){
        for(int j = 0; j < bn->size; j++){
            for (int k = 0; k < bn->size; k++){
                n = bn->nodes[i][j][k];
                if (n->parents == NULL){ 
                    n->parents = malloc(sizeof(Node) * n_incoming_relations);
                }
                n->n_parents = 0; 
                if (n->children == NULL){ 
                    n->children = malloc(sizeof(Node) * n_possibly_incoming_directions);
                }
                n->n_children = 0; 
            }
        }
    }
    
    for (int d = 0; d < depth; d++){
        if (verbose) printf("Adding relations for depth %d\n",d);

        for (int n_relation = 0; n_relation < n_incoming_relations; n_relation++){
            int best_relation_direction = -1;
            float best_heuristic = 10000000000000000;
            for(int direction  =0; direction < n_possibly_incoming_directions; direction++){
                
                if (heuristics[d][n_relation][direction] == 0.0){
                    printf("Error: missing value for heuristic of adding relation \n");
                    exit(1);
                }

                if (heuristics[d][n_relation][direction] < best_heuristic){
                    best_heuristic = heuristics[d][n_relation][direction];
                    best_relation_direction = direction;
                }
            }

            if (verbose) printf("Add incomming relations from depth %d from %d (%d) \n", best_relation_direction % depth, best_relation_direction / depth,best_relation_direction);

            //only child parent relations in order to parallelize this loop
            addRelationChildParent(bn,d,best_relation_direction,true);
        }
        if (verbose) {
            printf("Done for depth %d\n",d);
            printf("Example:\n");
            printNode(bn->nodes[d][bn->size / 2][bn->size / 2],true);
        }
    
    }

}

float *** initStructureHeuristic(int depth, int n_relations, int diagonal){
    int n_directions_relations = diagonal ? 4  * depth : 2 * depth;
    float *** structure_heuristic = malloc(sizeof(float **) * depth);
    for (int i = 0; i < depth; i++){
        structure_heuristic[i] = malloc(sizeof(float*) * n_relations);
        for(int j = 0; j < n_relations; j++){
            structure_heuristic[i][j] = malloc(sizeof(float) * n_directions_relations);
            for (int k = 0; k < n_directions_relations; k++){
                structure_heuristic[i][j][k] = 0.0;
            }
        }
    }
    return structure_heuristic;
}

void printStructureHeuristic(float *** structure_heuristic, int depth, int n_relations, int diagonal){
    int n_directions_relations = diagonal ? 4  * depth : 2 * depth;
    printf("Stucture heuristic with depth %d, n_relations = %d and n_directions = %d\n",depth,n_relations,n_directions_relations);
    for (int i = 0; i < depth; i++){
        printf("For incoming relations of nodes in depth %d:\n",i);
        for(int j = 0; j< n_relations; j++){
            printf("relation number %d: ",j);
            for(int k = 0; k < n_directions_relations; k++){
                printf("%.0f  ", structure_heuristic[i][j][k]);
            }
            printf("\n");
        }
    }
}

void freeStructureHeuristic(float *** structure_heuristic, int depth, int n_relations){
    for (int i = 0; i < depth; i++){
        for(int j = 0; j < n_relations; j++){
            free(structure_heuristic[i][j]);
        }
        free(structure_heuristic[i]);
    }
    free(structure_heuristic);
}

//update all values with 0.0 as well as all if building on previous 0.0 value
int  updateStructureHeuristics(float *** structureHeuristic,int n_relations,  BayesianNetwork bn, float **** data, int n_data,bool verbose){
    int  n_directions_relations = bn->diagonals ? 4  * bn->depth : 2 * bn->depth;
    int n_updates = 0;


    #pragma omp parallel for
    for (int d = 0; d < bn->depth; d++){

        float best_heuristic, current;
        int best_index;
        bool best_is_new, best_is_new_current;
        int * existing_relations = malloc(sizeof(int) * n_relations);
        bool already_exists_flag, is_new_flag;
        BayesianNetwork bn_copy;
        int data_used;

        if (verbose) printf("Updating for depth %d\n",d);

        bn_copy = createBayesianNetwork(bn->size,bn->depth,0,bn->distanceRelation,bn->diagonals);
        for(int x = 0; x < bn_copy->size; x++){
            for(int y = 0; y < bn_copy->size; y++){
                bn_copy->nodes[d][x][y]->parents = malloc(sizeof(Node) * n_relations);
            }
        }

        best_is_new = false;
        for (int n_relation = 0; n_relation < n_relations; n_relation++){
            if (verbose) printf("relation number  %d\n",n_relation);
            best_heuristic = 1000000000;
            best_index = -1;
            best_is_new_current = false;
            for(int direction = 0; direction < n_directions_relations; direction++){
                if (verbose) printf("direction  %d\n", direction);

                already_exists_flag = false;
                for(int i = 0; i < n_relation; i++){

                    if (existing_relations[i] == direction){
                        already_exists_flag = true;
                        break;
                    }
                }
                if (already_exists_flag){
                    structureHeuristic[d][n_relation][direction] = 80000000000000;
                    if (verbose) printf("... already exists\n");
                    continue;
                }

                current = structureHeuristic[d][n_relation][direction];
                is_new_flag = false;
                if (current == 0.0 || best_is_new){
                    if (verbose) printf("Must update because missing or previous was new\n");

                    data_used = (int)(pow(2,n_relation)) * STRUCTURE_SEARCH_DATA_PER_CONFIGURATION ;
                    if (data_used > n_data){
                        //printf("WARNING: it is not enough data instances per configuration\n");
                        data_used = n_data;
                    }

                    addRelationChildParent(bn_copy,d,direction,false);
                    current = bicOneLevel(bn_copy, data,data_used,d,false);
                    structureHeuristic[d][n_relation][direction] = current;
                    removeLastRelationChildParent(bn_copy,d,direction,false);

                    is_new_flag = true;
                    n_updates++;
                }
                if (current < best_heuristic){
                    best_heuristic = current;
                    best_index = direction;
                    best_is_new_current= is_new_flag;
                }
            }
            best_is_new = best_is_new_current ;
            if (best_is_new && n_relation != n_relations -1){
                if (verbose) printf("Must reset everything in next layer\n");
                //reset all values
                for(int direction = 0; direction < n_directions_relations; direction++){
                    structureHeuristic[d][n_relation+1][direction] = 0.0;
                }
            }
            existing_relations[n_relation] = best_index;
            addRelationChildParent(bn_copy,d,best_index,false);
        }
        freeBayesianNetwork(bn_copy);
        free(existing_relations);
    }
    return n_updates;
}

void learnStructureConvolutionalNodes(BayesianNetwork bn, int n_relations,float **** layered_images, int n_data){
    
    float *** structure_heuristics = initStructureHeuristic(bn->depth,n_relations,bn->diagonals);
    int n_updates = updateStructureHeuristics(structure_heuristics,n_relations,bn,layered_images,n_data,false);
    optimizeStructureUsingStructureHeuristic(bn,layered_images,n_data,structure_heuristics,n_relations,false);
    freeStructureHeuristic(structure_heuristics,bn->depth,n_relations);
}



void optimizeStructure(ConvolutionalBayesianNetwork cbn, int layer, int n_incoming_relations, float *** images, int * labels, int n_data, bool verbose){

    BayesianNetwork bn = cbn->bayesianNetworks[layer];
    //set data to data of layer
    float ****temp, **** layered_images = imagesToLayeredImagesContinuos(images,n_data,28);
    int d = 1,s = 28;
    for (int l = 0; l < layer+1; l++){
        temp = layered_images;
        layered_images = dataTransition(temp,n_data,d,s
            ,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

        freeLayeredImagesContinuos(temp,n_data,d,s);
        d = cbn->bayesianNetworks[l]->depth;
        s = cbn->bayesianNetworks[l]->size;
    }

    learnStructureNumberNodes(n_incoming_relations,layer,bn,layered_images,labels,n_data,verbose);
    learnStructureConvolutionalNodes(bn,n_incoming_relations,layered_images,n_data);

    freeLayeredImagesContinuos(layered_images,n_data,bn->depth,bn->size);
}

void addRandomStructure(ConvolutionalBayesianNetwork cbn, int layer, int n_incoming_relations){
    BayesianNetwork bn = cbn->bayesianNetworks[layer];

    if (n_incoming_relations > (bn->diagonals ? 4 : 2) * bn->depth){
        printf("ERROR: More than possible relations\n");
        exit(1);
    }

    int n_possibly_incoming_directions = bn->diagonals ? 4 * bn->depth : 2 * bn->depth;

    //remove all relations
    Node n;
    for (int i = 0; i < bn->depth; i++){
        for(int j = 0; j < bn->size; j++){
            for (int k = 0; k < bn->size; k++){
                n = bn->nodes[i][j][k];
                if (n->parents == NULL){ 
                    n->parents = malloc(sizeof(Node) * n_incoming_relations);
                }
                n->n_parents = 0; 
                if (n->children == NULL){ 
                    n->children = malloc(sizeof(Node) * n_possibly_incoming_directions);
                }
                n->n_children = 0; 
            }
        }
    }
    
    NumberNode nn;
    bool already_exists_flag;
    int direction;
    int * used_directions = malloc(sizeof(int) * n_incoming_relations);
    int parent_d, parent_y, parent_x;
    for (int d = 0; d < bn->depth; d++){
        for (int i = 0; i < n_incoming_relations; i++){
            used_directions[i] = -1;
        }

        for (int i = 0; i < n_incoming_relations; i++){
            already_exists_flag = true;
            while (already_exists_flag) {
                direction = rand() % ((bn->diagonals ? 4 : 2) * bn->depth);
                already_exists_flag = false;
                for (int j= 0; j < i; j++){
                    if (used_directions[j] == direction){
                        already_exists_flag = true;
                        break;
                    }
                }
            }
            used_directions[i] = direction;
            addRelationChildParent(bn,d,direction,true);
        }
    }

    for (int j = 0; j < bn->n_numberNodes; j++){
        nn = bn->numberNodes[j];
        if (nn->parents == NULL){
            nn->parents = malloc(sizeof(Node) * n_incoming_relations);
        }

        for (int i = 0; i < n_incoming_relations; i++){
            already_exists_flag = true;
            while (already_exists_flag) {
                parent_d = rand() % bn->depth;
                parent_x = rand() % bn->size;
                parent_y = rand() % bn->size;
                already_exists_flag = false;
                for (int k= 0; k < nn->n_parents; k++){
                    n = nn->parents[k];
                    if (n->depth == parent_d && n->x == parent_x && n->y == parent_y){
                        already_exists_flag = true;
                        break;
                    }
                }
            }
            n = bn->nodes[parent_d][parent_x][parent_y];
            nn->parents[nn->n_parents] = n;
            nn->n_parents++;

            n->numberNodeChildren = realloc(n->numberNodeChildren, sizeof(NumberNode) *  (n->n_numberNodeChildren + 1));
            n->numberNodeChildren[n->n_numberNodeChildren] = nn;
            n->n_numberNodeChildren++;
            
        }
    }

    free(used_directions);
}