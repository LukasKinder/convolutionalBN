
#define N_DATA_FOR_SELECTING_KERNEL 10 
#define N_DATA_FOR_FINDING_WORST_OR_NEW_KERNEL 1000
#define ITERATIONS_EM 3
#define STRUCTURE_SEARCH_DATA_PER_CONFIGURATION 10

float covariance(bool **** data, int n_data, int kernel1_depth,int kernel2_depth, int size, float mean1, float mean2){
    float covariance = 0;
    int i,j,k;
    for ( i  =0; i < n_data; i++){
        for (j = 0; j < size; j++){
            for (k = 0; k < size; k++){
                covariance += ((data[i][kernel1_depth][j][k] ? 1.0 :0.0) - mean1)
                            * ((data[i][kernel2_depth][j][k] ? 1.0 :0.0) - mean2);  
            }
        }
    }
    return covariance / (n_data * size * size);
}

float meanKernelValues(bool **** data, int n_data, int kernel_depth, int size){
    float mean = 0;
    int i,j,k;
    for ( i  =0; i < n_data; i++){
        for (j = 0; j < size; j++){
            for (k = 0; k < size; k++){
                mean += data[i][kernel_depth][j][k] ? 1.0 :0.0; 
            }
        }
    }
    return mean / (n_data * size * size);
}


float varianceKernelValues(bool **** data, int n_data, int kernel_depth, int size, float mean){

    int n_1 = (int)( (n_data *size * size) * mean);
    int n_0 = (n_data *size * size) -n_1;
    return ((1.0 - mean)*(1.0 - mean) * n_1 + (0.0 - mean)*(0.0 - mean)* n_0) / (n_data *size * size);
}

float correlation(bool **** data, int n_data, int kernel1_depth,int kernel2_depth, int size, float mean1, float mean2){

    float variance1 = varianceKernelValues(data,n_data,kernel1_depth,size,mean1);
    float variance2 = varianceKernelValues(data,n_data,kernel2_depth,size,mean2);

    if (variance1 == 0 || variance2 == 0) return 1.0; //Edge-case

    return covariance(data,n_data,kernel1_depth,kernel2_depth,size,mean1,mean2) 
            / (sqrt(variance1) * sqrt(variance2));
}

//replaces kernel with one that has a non 0 or 1 mean and a not too high correlation with other kernels
//"data_after" must contain the data_before transformed without pooling
void replace_kernel_with_promising_candidate(Kernel * kernels, int n_kernels, int index_kernel_to_replace
        ,bool **** data_before, int size_before, int n_data, bool verbose){

    n_data = n_data < N_DATA_FOR_FINDING_WORST_OR_NEW_KERNEL ? n_data: N_DATA_FOR_FINDING_WORST_OR_NEW_KERNEL;
    
    int max_iterations = 100;
    Kernel k = kernels[index_kernel_to_replace];
    int kernel_size = k.size;
    int kernel_depth = k.depth;
    int kernel_stride = k.stride;
    bool kernel_padding = k.padding;
    KernelType kernel_type =  k.type;
    float mean,corr;

    bool **** data_after = malloc(sizeof(bool***) * n_data);
    for(int i = 0; i < n_data; i++){
        data_after[i] = malloc(sizeof(bool **) * n_kernels);
        for(int j = 0; j < n_kernels; j++){
            data_after[i][j] = applyConvolution(data_before[i],size_before,size_before,kernels[j]);
        }
    }
    int size_after = sizeAfterConvolution(size_before,kernels[0]);

    float *means = malloc(sizeof(float) * n_kernels);
    for (int  i = 0; i < n_kernels; i++){
        if (i == index_kernel_to_replace) continue;
        means[i]  =meanKernelValues(data_after,n_data,i,size_after);
    }

    for (int i = 0; i < max_iterations; i++){

        freeKernel(k);

        if (verbose)  printf("REPLACE_WITH_PROMISING_KERNEL: Iteration %d / %d: ",i,max_iterations);

        k = createKernel(kernel_size, kernel_depth ,kernel_type, kernel_stride,kernel_padding);

        //free the data of the old kernel and replace it with the new one
        for (int j = 0; j < n_data; j++){
            freeImage(data_after[j][index_kernel_to_replace],size_after);
            data_after[j][index_kernel_to_replace] = applyConvolution(data_before[j],size_before,size_before,k);
        }
        mean = meanKernelValues(data_after,n_data,index_kernel_to_replace,size_after);
        if (mean < 0.0003 || mean > 0.999){
            //kernel is not interesting enough
            if (verbose)  printf("REPLACE_WITH_PROMISING_KERNEL: Kernel is not interesting enough (mean = %f)\n",mean);
            continue;
        }

        for (int j = 0; j < n_kernels; j++){
            if (j == index_kernel_to_replace || means[j] == 0.0) continue;
            corr = fabs( correlation(data_after,n_data,index_kernel_to_replace,j,size_after,mean,means[j]));
            if (0.4 < corr ){
                //kernel is too similar to an existing kernel
                if (verbose)  printf("REPLACE_WITH_PROMISING_KERNEL: Kernel is too similar to an existing one (%d ==> %f\n",j,corr);
                break;
            }
        }
        if ( 0.4 < corr) continue;

        if (verbose)  printf("REPLACE_WITH_PROMISING_KERNEL: Found a useful kernel! (mean = %f, corr = %f)\n", mean,corr);

        //kernel is meaningful and not too similar to an existing one
        kernels[index_kernel_to_replace] = k;
        break;
    }

    free(means);
    freeLayeredImages(data_after,n_data,n_kernels,size_after);
}

void initKernels(ConvolutionalBayesianNetwork cbn, int layer
        ,bool **** data_before, int n_data, bool verbose){

    if (verbose) printf("INIT_KERNELS: Init variables\n");
    int data_used = n_data < N_DATA_FOR_SELECTING_KERNEL ? n_data : N_DATA_FOR_SELECTING_KERNEL;
    int n_kernels = cbn->n_kernels[layer-1];
    Kernel * kernels = cbn->transitionalKernels[layer-1];
    int size_before = cbn->bayesianNetworks[layer-1]->size;
    int size_after = sizeAfterConvolution(size_before,kernels[0]);

    bool **** newData = malloc(sizeof(bool***) * data_used);
    if (verbose) printf("INIT_KERNELS: Init testing data\n");
    #pragma omp parallel for
    for (int  i = 0; i < data_used; i++){
        newData[i] = malloc(sizeof(bool **) * n_kernels);

        #pragma omp parallel for
        for (int j = 0; j < n_kernels;j++){
            newData[i][j] = applyConvolution(data_before[i],size_before,size_before,kernels[j]);
        }
    }
    for (int j = 0; j < n_kernels;j++){
        if (verbose) printf("INIT_KERNELS: Init kernel %d (size after = %d)\n",j,size_after);
        replace_kernel_with_promising_candidate(kernels,n_kernels,j,data_before,size_before
                ,data_used,verbose);
    }

    freeLayeredImages(newData,data_used,n_kernels,size_after);
}


//does only add add parent->child relations if "add_child_relation" = true!
// if diagonals = false: order  = > up, left
// if diagonals = true:  order = > up-left, up, up-right, left
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



/*
previous gains (0.0 = missing value) (10000000000000001 = NA):
    For each Kernel:
        After adding i'ths relation
            bic gain after adding relation
*/
bool optimizeStructure(BayesianNetwork bn,  bool **** data, int n_data
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

void updateStructureHeuristicNewKernel(float *** structure_heuristic,int depth, int n_relations, int diagonal, int kernel, bool verbose){
    int n_directions_relations = diagonal ? 4  * depth : 2 * depth; 
    float bestHeuristic, current;
    int best_index;
    bool best_was_updated;

    int n_to_zero = 0;

    for (int d = 0; d < depth; d++){

        best_was_updated = false;
        for (int n_relation = 0; n_relation < n_relations; n_relation++){

            if (! best_was_updated){
                bestHeuristic = 1000000000;
                for (int direction =0; direction < n_directions_relations; direction++){
                    current  = structure_heuristic[d][n_relation][direction];
                    if (current < bestHeuristic){
                        bestHeuristic = current;
                        best_index = direction;
                    }
                }
            }
            
            for (int direction = 0; direction < n_directions_relations; direction++){
                if (best_was_updated || direction % depth == kernel){
                    structure_heuristic[d][n_relation][direction] = 0.0;
                    n_to_zero++;
                }
            }

            if (! best_was_updated){
                //check if the best was updated, in this case, everything must be reset from now one
                if (best_index  % depth == kernel){
                    //conecutive n_relatiosn need a full reset
                    best_was_updated = true;
                }
            }
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
int  updateStructureHeuristics(float *** structureHeuristic,int n_relations,  BayesianNetwork bn, bool **** data, int n_data,bool verbose){
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

        bn_copy = createBayesianNetwork(bn->size,bn->depth,bn->distanceRelation,bn->diagonals);
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
                    current = bicOneLevel(bn_copy, data,data_used,d,false );
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

//heuristic is log liklyhood after removeing relations (should be higher)
float kernelHeuristic(bool **** data, int n_data, BayesianNetwork bn, int depth_kernel, float * logProbLevel, bool verbose){

    BayesianNetwork bn_no_outgoing = createBayesianNetwork(bn->size,bn->depth,bn->distanceRelation, bn->diagonals);
    int d,x,y, n_child;
    Node original_node, original_child,new_node,new_child;
    bool * relevant_layer = malloc(sizeof(bool) * bn->depth);

    if (verbose){
        printf("removing outgoing relations of kernel %d has an effect on levels: ",depth_kernel);
    }
    for(int i = 0; i < bn->depth; i++){
        relevant_layer[i] = false;
    }

    //same relations as bn but nodes at depth_kernel do not have outgoing relations
    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (int y = 0; y < bn->size; y ++){
                if (d == depth_kernel){
                    //do not add outgoing relations for this kernel!
                    original_node = bn->nodes[d][x][y];
                    for (n_child = 0; n_child < original_node->n_children; n_child++){
                        original_child = original_node->children[n_child];
                        if (! relevant_layer[original_child->depth] && verbose) printf("%d ", original_child->depth);
                        relevant_layer[original_child->depth] = true; //this layer is important!
                    }
                    continue;
                }

                original_node = bn->nodes[d][x][y];
                new_node = bn_no_outgoing->nodes[original_node ->depth][original_node ->x][original_node ->y];

                for (n_child = 0; n_child < original_node->n_children; n_child++){
                    original_child = original_node->children[n_child];
                    new_child = bn_no_outgoing->nodes[original_child->depth][original_child->x][original_child->y];
                    if (new_child->parents == NULL){
                        //might be smaller than this, but better than re-allocating
                        new_child->parents = malloc(sizeof(Node) * original_child->n_parents);
                    }
                    //add relations
                    new_child->parents[new_child->n_parents] = new_node;
                    new_child->n_parents++;
                }
            }
        }
    }
    if (verbose) printf("\n Initialization complete \n");


    //the actual heuristic, removing outgoin relations should reduce the likleyhood. Effect (probably) bigger if kernel is more important
    float heuristic = 0;
    for( int i = 0; i< bn->depth; i++){
        if (relevant_layer[i]){
            heuristic += logMaxLikelihoodDataGivenModelOneLevel(bn_no_outgoing,data,n_data,i);
            if (verbose) printf("in layer  %d, heuristic += %f \n", i, logMaxLikelihoodDataGivenModelOneLevel(bn_no_outgoing,data,n_data,i));
            
        } else{
            //reuse previous results
            heuristic += logProbLevel[i];
        }
    }

    if (verbose) printf("overall heuristic %f\n",heuristic);

    free(relevant_layer);

    freeBayesianNetwork(bn_no_outgoing);
    return heuristic;

}

int determine_worst_kernel(bool **** data, int n_data, BayesianNetwork bn, bool verbose){
    float * logProbsLevel = malloc(sizeof(float) * bn->depth);

    int used_data = N_DATA_FOR_FINDING_WORST_OR_NEW_KERNEL < n_data ? N_DATA_FOR_FINDING_WORST_OR_NEW_KERNEL : n_data;

    for(int i = 0; i < bn->depth; i++){
        logProbsLevel[i] = logMaxLikelihoodDataGivenModelOneLevel(bn,data,used_data,i);
    }
    int worst_kernel = -1;
    float current, worst_heuristic = -999999999999;
    #pragma omp parallel for
    for (int i = 0; i < bn->depth; i++){
        if (verbose){
            printf("finding heuristic for kernel %d\n",i);
        }
        current = kernelHeuristic(data,used_data,bn,i,logProbsLevel, verbose);
        #pragma omp critical
        {
        if (current > worst_heuristic){
            worst_heuristic  = current;
            worst_kernel = i;
        }
        }

    }
    free(logProbsLevel);

    return worst_kernel;
}

void optimizeKernelsAndStructure(ConvolutionalBayesianNetwork cbn, int layer, bool *** images,int n_data, int n_relations, bool verbose){

    bool ****temp, **** layered_images = imagesToLayeredImages(images,n_data,28);
    int d,s;
    if ( verbose) printf("OPTIMIZE_KERNELS_AND_STRUCTURE: Gernerate data at layer before\n");
    for (int l = 0; l < layer-1; l++){
        temp = layered_images;
        d = cbn->bayesianNetworks[l]->depth;
        s = cbn->bayesianNetworks[l]->size;
        layered_images = dataTransition(temp,n_data,d,s
            ,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

        freeLayeredImages(temp,n_data,d,s);
    }
    if ( verbose) printf("OPTIMIZE_KERNELS_AND_STRUCTURE: init Kernels\n");

    BayesianNetwork bn = cbn->bayesianNetworks[layer];
    BayesianNetwork bn_before = cbn->bayesianNetworks[layer-1];
    initKernels(cbn,layer,layered_images,n_data,verbose);

    if (n_relations == 0){
        printf("WARNING: 0 relations!");
        return;
    }


    bool **** data_next_layer;
    data_next_layer = dataTransition(layered_images, n_data,bn_before->depth,bn_before->size,cbn->transitionalKernels[layer-1]
                ,cbn->n_kernels[layer-1],cbn->poolingKernels[layer-1] );


    int index_worst_kernel;
    float *** structure_heuristics = initStructureHeuristic(bn->depth,n_relations,bn->diagonals);
    int n_updates;
    bool ** tmp;
    int size_after;

    //add child-parent-arrays to nodes with correct size

    for (int i = 0; i < ITERATIONS_EM ; i++){

        if (verbose) printf("EM_ iteration %d / %d\n",i+1, ITERATIONS_EM);
        
        if (verbose) printf("Optimizing Structure\n");

        //printStructureHeuristic(structure_heuristics,bn->depth,n_relations,bn->diagonals);
        n_updates = updateStructureHeuristics(structure_heuristics,n_relations,bn,data_next_layer,n_data,false);
        if (verbose) printf("...required %d updates\n",n_updates);


        //printStructureHeuristic(structure_heuristics,bn->depth,n_relations,bn->diagonals);
        optimizeStructure(bn,data_next_layer,n_data,structure_heuristics,n_relations,false);

        if (verbose) printf("Finding worst kernel\n");

        index_worst_kernel = determine_worst_kernel(data_next_layer,n_data,bn,false);

        if (verbose) printf("REplacing worst kernel (%d)\n",index_worst_kernel);

        replace_kernel_with_promising_candidate(cbn->transitionalKernels[layer-1],cbn->n_kernels[layer-1]
            ,index_worst_kernel,layered_images,cbn->bayesianNetworks[layer-1]->size,n_data,false);

        if (verbose) printf("Update Structure Heuristic\n");

        updateStructureHeuristicNewKernel(structure_heuristics,bn->depth,n_relations,bn->diagonals,index_worst_kernel,false);

        if (verbose) printf("Update data\n");
        
        for (int j = 0; j < n_data; j++){
            freeImage(data_next_layer[j][index_worst_kernel],bn->size);
            tmp = applyConvolution(layered_images[j],bn_before->size,bn_before->size,cbn->transitionalKernels[layer-1][index_worst_kernel]);
            size_after = sizeAfterConvolution(bn_before->size,cbn->transitionalKernels[layer-1][index_worst_kernel]);
            data_next_layer[j][index_worst_kernel] = applyMaxPoolingOneLayer(tmp,size_after,size_after,cbn->poolingKernels[layer-1]);
            freeImage(tmp,size_after);
        }
    }
    freeStructureHeuristic(structure_heuristics,bn->depth,n_relations);
    freeLayeredImages(data_next_layer,n_data,bn->depth,bn->size);
    d = cbn->bayesianNetworks[layer-1]->depth;
    s = cbn->bayesianNetworks[layer-1]->size;
    freeLayeredImages(layered_images,n_data,d,s);
}