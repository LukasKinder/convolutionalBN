
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
//TODO use different search methods
void optimizeStructure(BayesianNetwork bn,  bool **** data, int n_data, bool diagonal){

    if (diagonal){
        printf("diagonal relations not implemented\n");
        exit(1);
    }

    int d = bn->depth;
    bool** upRelations = malloc(sizeof(bool*) * d);
    bool** leftRelations = malloc(sizeof(bool*) * d);

    int iterations = 3;
    int change_index = -1;

    float heuristic, new_heuristic;

    for (int i = 0; i < d; i++){

        printf("Optimizing layer %d\n",i);

        upRelations[i] = malloc(sizeof(bool) * d);
        leftRelations[i] = malloc(sizeof(bool) * d);

        heuristic = bicOneLevel(bn,data,n_data,i);

        for (int j = 0; j < d; j++){
            upRelations[i][j] = false;
            leftRelations[i][j] = false;
        }

        for (int iteration = 0 ; iteration < iterations; iteration++){

            printf("iteration %d\n",iteration);

            change_index = rand() % d;
            if (iteration %2 == 0){
                upRelations[i][change_index] = !upRelations[i][change_index];
            }else{
                leftRelations[i][change_index]  = !leftRelations[i][change_index];
            }

            printf("mutated relation array\n");

            updateParentsLevel(bn,i,upRelations[i],leftRelations[i]);
            printf("updated relations\n");

            new_heuristic = bicOneLevel(bn,data,n_data,i);
            if (new_heuristic < heuristic){
                printf("Suggested change is accepted\n");
                heuristic = new_heuristic;
            }else{
                printf("Suggested change is rejected\n");
                if (iteration %2 == 0){
                    upRelations[i][change_index] = !upRelations[i][change_index];
                }else{
                    leftRelations[i][change_index]  = !leftRelations[i][change_index];
                }
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
void optimizeKernels(BayesianNetwork bn, Kernel * kernels, int n_kernels, Kernel poolingKernel, bool **** data){

}