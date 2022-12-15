

#define MAX_ITERATIONS_EM 2 //default 5
#define KERNEL_SEARCH_ITERATION_PER_PARAMETER 2 //default 10
#define MAX_ITERATIONS_STRUCTURE_SEARCH 10 //default 15
#define KERNEL_OPTIMIZATION_LOCAL_SEARCH false



void addParentChildRelations(BayesianNetwork bn,bool** upRelations, bool ** leftRelations){
    Node n,parent;
    for(int i = 0; i < bn->depth; i++){
        for (int j = 0; j < bn->size; j++){
            for (int k = 0; k < bn->size; k++){
                n = bn->nodes[i][j][k];
                if (n->parents == NULL) n->parents = malloc(sizeof(Node) * bn->depth * 2);
                if (n->children == NULL) n->children = malloc(sizeof(Node) * bn->depth * 2);
                n->n_parents = 0;
                n->n_children = 0;
            }
        }
    }

    for(int i = 0; i < bn->depth; i++){
        for (int j = 0; j < bn->size; j++){
            for (int k = 0; k < bn->size; k++){
                n = bn->nodes[i][j][k];
                for (int parent_d = 0; parent_d < bn->depth; parent_d++){
                    if (upRelations[i][parent_d] && n->y - bn->distanceRelation >= 0){
                        parent = bn->nodes[parent_d][n->x][n->y - bn->distanceRelation];
                        n->parents[n->n_parents] = parent;
                        (n->n_parents)++;
                        parent->children[parent->n_children] = n;
                        (parent->n_children)++;
                    }
                    if (leftRelations[i][parent_d] && n->x - bn->distanceRelation >= 0){
                        parent = bn->nodes[parent_d][n->x - bn->distanceRelation ][n->y];
                        n->parents[n->n_parents] = parent;
                        (n->n_parents)++;
                        parent->children[parent->n_children] = n;
                        (parent->n_children)++;
                    }
                }
            }
        }
    }
}

//does NOT update children
void updateParentsLevel(BayesianNetwork bn,int level, bool * upRelations, bool * leftRelations){
    Node n;
    for (int x = 0; x < bn->size; x++){
        for (int y = 0; y < bn->size; y++){
            n = bn->nodes[level][x][y];
            if (n->parents == NULL) n->parents = malloc(sizeof(Node) * bn->depth * 2);
            n->n_parents = 0;


            for (int parent_d = 0; parent_d < bn->depth; parent_d++){
                if (upRelations[parent_d] && n->y - bn->distanceRelation >= 0){
                    n->parents[n->n_parents] = bn->nodes[parent_d][n->x][n->y - bn->distanceRelation];
                    (n->n_parents)++;
                }
                if (leftRelations[parent_d] && n->x - bn->distanceRelation >= 0){
                    n->parents[n->n_parents] = bn->nodes[parent_d][n->x - bn->distanceRelation][n->y];
                    (n->n_parents)++;
                }
            }
        }
    }
}

//optimizes the structure of the BN given the data of the BN
//TODO use different search methods, maybe SA
//TODO optimize by using relusts of previous iteration!!!
bool optimizeStructure(BayesianNetwork bn,  bool **** data, int n_data, bool diagonal, bool verbose){

    if (verbose) printf("Optimizing structure \n");

    if (diagonal){
        printf("diagonal relations not implemented\n");
        exit(1);
    }

    bool change = false;
    int d = bn->depth;
    bool** upRelations = malloc(sizeof(bool*) * d);
    bool** leftRelations = malloc(sizeof(bool*) * d);


    #pragma omp parallel for
    for (int i = 0; i < d; i++){

        if (verbose) printf("Optimizing Structure level %d / %d \n",i,d);

        float current_heuristic, best_heuristic,h;
        int best_index;
        bool bestIsUp;


        upRelations[i] = malloc(sizeof(bool) * d);
        leftRelations[i] = malloc(sizeof(bool) * d);

        for (int j = 0; j < d; j++){
            upRelations[i][j] = false;
            leftRelations[i][j] = false;
        }
        updateParentsLevel(bn,i,upRelations[i],leftRelations[i]);

        current_heuristic = bicOneLevel(bn,data,n_data,i,false);

        // add relations one by one
        for (int iteration = 0 ; iteration < MAX_ITERATIONS_STRUCTURE_SEARCH ; iteration++){

            if (verbose) printf("Iteration %d: ",iteration);

            best_heuristic = 10000000000000000;

            for (int j = 0; j < d; j++){
                if (! upRelations[i][j]){
                    upRelations[i][j] = true;
                    updateParentsLevel(bn,i,upRelations[i],leftRelations[i]);
                    h= bicOneLevel(bn,data,n_data,i,false);
                    if (h < best_heuristic){
                        best_heuristic = h;
                        best_index = j;
                        bestIsUp = true;
                    }
                    upRelations[i][j] = false;
                }
                if (! leftRelations[i][j]){
                    leftRelations[i][j] = true;
                    updateParentsLevel(bn,i,upRelations[i],leftRelations[i]);
                    h= bicOneLevel(bn,data,n_data,i,false);
                    if (h < best_heuristic){
                        best_heuristic = h;
                        best_index = j;
                        bestIsUp = false;
                    }
                    leftRelations[i][j] = false;
                }
            }            
            

            if (best_heuristic < current_heuristic){
                if (verbose) printf("Found improvement; bic %.0f === %.0f \n",current_heuristic,best_heuristic);
                //commit the change
                if (bestIsUp){
                    upRelations[i][best_index] = true;
                }else{
                    leftRelations[i][best_index] = true;
                }
                current_heuristic = best_heuristic;
                updateParentsLevel(bn,i,upRelations[i],leftRelations[i]);
            }else{
                //can not reach improvement using hill climbing
                updateParentsLevel(bn,i,upRelations[i],leftRelations[i]);
                if (verbose) printf("No improvement (best was %.0f ===> %.0f) - break  \n",current_heuristic,best_heuristic);
                break;
            }
        }
    }

    addParentChildRelations(bn,upRelations,leftRelations);

    for (int i = 0; i < d; i++){
        free(upRelations[i]);
        free(leftRelations[i]);
    }

    free(upRelations);
    free(leftRelations);

}

//ToDo: can be made more efficiant by not creating a copy of the bn
//Important that bn has updated counts!
float kernelHeuristic(bool **** data, int n_data, BayesianNetwork bn, int depth_kernel){
    BayesianNetwork bn_no_outgoing = createBayesianNetwork(bn->size,bn->depth,bn->distanceRelation);
    int d,x,y, n_child;
    Node original_node, original_child,new_node,new_child;

    //same relations as bn but nodes at depth_kernel do not have outgoing relations
    for (d = 0; d < bn->depth; d++){
        if (d == depth_kernel) continue; //do not add outgoing relations for nodes in this layer
        for (x = 0; x < bn->size; x++){
            for (int y = 0; y < bn->size; y ++){
                original_node = bn->nodes[d][x][y];
                new_node = bn_no_outgoing->nodes[original_node ->depth][original_node ->x][original_node ->y];
                new_node->children = malloc(sizeof(Node) * original_node->n_children);

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
                    new_node->children[new_node->n_children] = new_child;
                    new_node->n_children;
                }
            }
        }
    }
    //the actual heuristic, removing outgoin relations should reduce the bic. Effect (probably) bigger if kernel is more important
    float bic_improvement = bic(bn_no_outgoing,data,n_data,true,false) - bic(bn,data,n_data,false,false);

    freeBayesianNetwork(bn_no_outgoing);
    return bic_improvement;

}

//TODO: maybe replace multiple kernels at once
//is changing data_after!
void optimizeKernelsTryOut(BayesianNetwork bn, Kernel * kernels, int n_kernels, Kernel poolingKernel
        ,bool **** data_before, int size_before ,bool **** data_after, int n_data, bool verbose){

    //ToDo: maybe tune
    int max_iterations = 1000;
    int n_used = (n_data < 20 ? n_data: 20);

    fitDataCounts(bn,data_after,n_data);
    int i,j,k;
    int worstKernel = -1;
    float current_heuristic, worst_heuristic = 999999999;
    if (verbose) printf("Optimizing kernels by replacing the worst\n");

    for (int i  =0; i < n_kernels; i++){
        current_heuristic = kernelHeuristic(data_after,n_data,bn,i);

        if (verbose) printf("Kernel %d has a heuristic of %f\n",i,current_heuristic);

        if (current_heuristic < worst_heuristic){
            worstKernel = i;
            worst_heuristic = current_heuristic;
        }
    }
    if (verbose) printf("Worst kernel is number %d with a heuristic of %f\n",worstKernel,worst_heuristic);
    int kernel_size = kernels[worstKernel].size;
    int kernel_depth = kernels[worstKernel].depth;
    int kernel_stride = kernels[worstKernel].stride;
    bool kernel_padding = kernels[worstKernel].padding;
    KernelType kernel_type =  kernels[worstKernel].type;

    bool **tmp;
    int tmp_size = sizeAfterConvolution(size_before,kernels[worstKernel]);
    freeKernel(kernels[worstKernel]);

    if (verbose) printf("Searching for new promising kernel\n");

    float *means = malloc(sizeof(float) * n_kernels);
    for (int  i = 0; i < n_kernels; i++){
        if (i == worstKernel) continue;
        means[i]  =meanKernelValues(data_after,n_used,i,bn->size);
    }


    Kernel new_kernel;
    float mean,  corr;
    poolingKernel.depth = 1;
    for (i = 0; i < max_iterations; i++){

        if (verbose)  printf("Iteration %d / %d: ",i,max_iterations);

        new_kernel = createKernel(kernel_size, kernel_depth ,kernel_type, kernel_stride,kernel_padding);

        //free the data of the old kernel
        for (j = 0; j < n_used; j++){
            freeImage(data_after[j][worstKernel],bn->size);
            tmp = applyConvolution(data_before[j],size_before,size_before,new_kernel);

            data_after[j][worstKernel] = applyMaxPoolingOneLayer(tmp,tmp_size,tmp_size,poolingKernel);
            freeImage(tmp,tmp_size);
        }
        mean = meanKernelValues(data_after,n_used,worstKernel,bn->size);
        if (mean < 0.01 || mean > 0.99){
            //kernel is not interesting enough
            //ToDo overwrite instead of free/create maybe
            freeKernel(new_kernel);
            if (verbose)  printf("Kernel is not interesting enough (mean = %f)\n",mean);
            continue;
        }

        for (int j = 0; j < n_kernels; j++){
            if (j == worstKernel) continue;
            corr = fabs( correlation(data_after,n_used,worstKernel,j,bn->size,mean,means[j]));
            if (0.3 < corr ){
                //kernel is too similar to an existing kernel
                if (verbose)  printf("Kernel is too similar to an existing one (%d ==> %f\n",j,corr);
                freeKernel(new_kernel);
                break;
            }
        }
        if ( 0.3 < corr) continue;

        if (verbose)  printf("Found a useful kernel! (mean = %f, corr = %f)\n", mean,corr);

        //kernel is meaningful and not too similar to an existing one
        kernels[worstKernel] = new_kernel;
        break;
    }
    poolingKernel.depth = n_kernels;

    free(means);

}

//optimizes the kernels given the data in the previous layer and a bayesian network with structure
//TODO different search methods, for now strict climbing
//TODO for all types of kernels, for now only mustInMustOutEither
//assumes that kernels are initialized in a good way
bool optimizeKernels(BayesianNetwork bn, Kernel * kernels, int n_kernels
        , Kernel poolingKernel, bool **** data, int n_data, int size_data, bool verbose){

    if (verbose) printf("Optimizing Kernels\n");
    
    BayesianNetwork bn_no_relations = createBayesianNetwork(bn->size,bn->depth,bn->distanceRelation);

    //number of iterations is 10 times the amount of parameters
    int n_iterations = (int) (KERNEL_SEARCH_ITERATION_PER_PARAMETER 
            * n_kernels * kernels[0].depth * kernels[0].size * kernels[0].size);

    Kernel randomKernel;
    int randD,randX,randY, index_rand_kernel;
    MT_MF_E_Value previous;

    bool **** newData;

    int depth_previous = kernels[0].depth;
    int size_kernels = kernels[0].size;

    int size_after_normal_kernels = sizeAfterConvolution(size_data,kernels[0]);
    int size_after = sizeAfterConvolution(size_after_normal_kernels, poolingKernel);

    newData = dataTransition(data,n_data,depth_previous,size_data,kernels,n_kernels,poolingKernel);

    float new_bic_improvement, current_bic_improvement;
    current_bic_improvement =  bic(bn,newData,n_data,true,false) - bic(bn_no_relations,newData,n_data,true,false);

    if (verbose) printf("initialization done\n");

    for (int iteration = 0; iteration < n_iterations; iteration++){
        index_rand_kernel = rand() % n_kernels;
        randomKernel = kernels[index_rand_kernel];
        randD = rand() % depth_previous;
        randX = rand() % size_kernels;
        randY = rand() % size_kernels;
        previous = randomKernel.map[randD][randX][randY];
        while (randomKernel.map[randD][randX][randY] == previous) {
            switch (rand() % 3){
            case 0:
                randomKernel.map[randD][randX][randY] = must_true;
                break;
            case 1:
                randomKernel.map[randD][randX][randY] = must_false;
                break;
            case 2:
                randomKernel.map[randD][randX][randY] = either;
                break;
            }
        }


        #pragma omp parallel for
        for(int i =0; i < n_data; i++){
            freeImage(newData[i][index_rand_kernel],size_after);
            bool ** intermediate_result  = applyConvolution(data[i],size_data,size_data,randomKernel);
            newData[i][index_rand_kernel] = applyMaxPoolingOneLayer(intermediate_result,size_after_normal_kernels,size_after_normal_kernels,poolingKernel);
            freeImage(intermediate_result,size_after_normal_kernels);
        }
        
        new_bic_improvement = bic(bn,newData,n_data,true,false) - bic(bn_no_relations,newData,n_data,true,false);
        

        if (verbose  && iteration% 30 == 0) printf("Iteration %d / %d current  bic: %f, suggested bic: %f",iteration,n_iterations,current_bic_improvement,new_bic_improvement);

        if (new_bic_improvement <= current_bic_improvement){
            current_bic_improvement = new_bic_improvement ;
            //keep change
            if (verbose  && iteration% 30 == 0) printf("  ===> accepted\n");
        }else{
            //discard Change
            randomKernel.map[randD][randX][randY] = previous;
            if (verbose  && iteration% 30 == 0) printf("  ===> rejected\n");
        }
    }

    freeBayesianNetwork(bn_no_relations);
    freeLayeredImages(newData,n_data,n_kernels,size_after);

    if (verbose) printf("Done\n");
}


void em_algorithm(BayesianNetwork bn, Kernel * kernels, int n_kernels, Kernel poolingKernel, bool **** data, int n_data, int size_data){

    pretrainKernels(kernels,n_kernels,poolingKernel,data,n_data,size_data,bn->distanceRelation,true);
    bool **** data_next_layer = dataTransition(data,n_data,kernels[0].depth,size_data,kernels,n_kernels,poolingKernel);

    float current_bic = bic(bn,data_next_layer,n_data,true,false);
    float next_bic;

    printf("Running EM algorithm\n");

    for (int i = 0; i < MAX_ITERATIONS_EM ; i++){
        
        printf("Optimizing Structure\n");
        optimizeStructure(bn,data_next_layer,n_data,false,false);

        if (KERNEL_OPTIMIZATION_LOCAL_SEARCH){
            optimizeKernels(bn,kernels,n_kernels,poolingKernel,data,n_data,size_data,false);
        }else{
            printf("Adapting Kernels\n");
            optimizeKernelsTryOut(bn,kernels,n_kernels, poolingKernel ,data,size_data,data_next_layer,n_data,true);
        }
        freeLayeredImages(data_next_layer,n_data,bn->depth,bn->size);
        data_next_layer = dataTransition(data,n_data,kernels[0].depth,size_data,kernels,n_kernels,poolingKernel);
        next_bic = bic(bn,data_next_layer,n_data,true,false);

        printf("bic reduction: %.1f ==> %.1f \n",current_bic,next_bic);
        /*if ( next_bic >= current_bic ){
            break;
        }*/
        current_bic = next_bic;
    }

    printf("Found Kernels:\n");
    for (int i = 0; i < n_kernels;i++){
        printKernel(kernels[i]);
    }    

    freeLayeredImages(data_next_layer,n_data,bn->depth,bn->size);
}