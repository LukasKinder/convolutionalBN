

#define MAX_ITERATIONS_EM 4 //default 5
#define KERNEL_SEARCH_ITERATION_PER_PARAMETER 4 //default 10
#define MAX_ITERATIONS_KERNEL_SEARCH 10 //default 15



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
        for (int iteration = 0 ; iteration < MAX_ITERATIONS_KERNEL_SEARCH ; iteration++){

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
    printf("depth previous %d; size kernels = %d\n",depth_previous,size_kernels);

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
            bool ** intermediate_result  = applyConvolution(data[i],size_data,randomKernel);
            newData[i][index_rand_kernel] = applyMaxPoolingOneLayer(intermediate_result,size_after_normal_kernels,poolingKernel);
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
    
    bool **** data_next_layer = dataTransition(data,n_data,kernels[0].depth,size_data,kernels,n_kernels,poolingKernel);
    float current_bic = bic(bn,data_next_layer,n_data,true,false);
    float next_bic;

    pretrainKernels(kernels,n_kernels,poolingKernel,data,n_data,size_data,bn->distanceRelation,true);


    printf("Running EM algorithm\n");

    for (int i = 0; i < MAX_ITERATIONS_EM ; i++){

        optimizeStructure(bn,data_next_layer,n_data,false,true);
        freeLayeredImages(data_next_layer,n_data,bn->depth,bn->size);

        optimizeKernels(bn,kernels,n_kernels,poolingKernel,data,n_data,size_data,true);

        data_next_layer = dataTransition(data,n_data,kernels[0].depth,size_data,kernels,n_kernels,poolingKernel);
        next_bic = bic(bn,data_next_layer,n_data,true,false);

        printf("bic reduction: %.1f ==> %.1f \n",current_bic,next_bic);
        if ( next_bic >= current_bic ){
            break;
        }
        current_bic = next_bic;
    }

    printf("Found Kernels:\n");
    for (int i = 0; i < n_kernels;i++){
        printKernel(kernels[i]);
    }    

    freeLayeredImages(data_next_layer,n_data,bn->depth,bn->size);
}