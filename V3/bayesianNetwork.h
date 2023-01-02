

typedef struct RawNode *Node;
typedef struct RawNumberNode *NumberNode;

typedef struct RawNode {
    bool value;

    //position within bn grid
    int x;
    int y;
    int depth;

    Node* parents;
    Node* children;
    int n_parents;
    int n_children;

    NumberNode * numberNodeChildren;
    int n_numberNodeChildren;

    int changeID; //id that is set when the value of a node is changes.

    int* stateCountsTrue; // maps parent configuration as index (binary number) to counts to be true
    int* stateCountsFalse; // maps parent configuration as index (binary number) to counts to be false
    float* CPT; // maps parent configuration as index (binary number) to probability to be true
  
}RawNode;


typedef struct RawNumberNode {
    int value;
    Node* parents;
    int n_parents;
    int changeID; //id that is set when the value of a node is changes.
    int** stateCounts; //rows are parent configurations, the 10 couloumns are the counts for its state
    float** CPT; //rows are parent configurations, the 10 couloumns are the probs for each state (rows add up to 1)
}RawNumberNode;

typedef struct RawBayesianNetwork *BayesianNetwork;

typedef struct RawBayesianNetwork {
    Node*** nodes; //nodes within the x,y,depth grid
    int size; //size in terms of x,y
    int depth; //size in terms of depth
    int distanceRelation;
    bool diagonals; //wether to include diagonal relations

    NumberNode * numberNodes;
    int n_numberNodes;
}RawBayesianNetwork;

Node initNode(int depth, int x, int y){
    Node n = malloc(sizeof(RawNode));
    n->parents = NULL;
    n->children = NULL;
    n->stateCountsTrue = NULL;
    n->stateCountsFalse = NULL;
    n->CPT = NULL;
    n->depth = depth;
    n->x = x;
    n->y = y;
    n->n_parents = 0;
    n->n_children = 0;
    n->changeID = 0;

    n->numberNodeChildren = NULL;
    n->n_numberNodeChildren = 0;
    return n; 
}

NumberNode initNumberNode(){
    NumberNode nn = malloc(sizeof(RawNumberNode));
    nn->parents = NULL;
    nn->n_parents = 0;
    nn->value = 1; //HARDCODED!!!
    nn->changeID = 0;
    nn->CPT = NULL;
    nn->stateCounts = NULL;
    nn->changeID = 0;
    return nn; 
}

BayesianNetwork createBayesianNetwork(int size, int depth, int distance_relation, bool diagonals){
    BayesianNetwork bn = malloc(sizeof(RawBayesianNetwork));
    bn->size = size;
    bn->diagonals = diagonals;
    bn->distanceRelation = distance_relation;
    int i,j,k;
    bn->depth = depth;
    bn->nodes = malloc(sizeof(Node**) * depth);
    for (i = 0; i < depth; i++){
        bn->nodes[i] = malloc(sizeof(Node*) * size);
        for (j = 0; j < size; j++){
            bn->nodes[i][j] = malloc(sizeof(Node) * size);
            for (k = 0; k < size; k++){
                bn->nodes[i][j][k] = initNode(i,j,k);
            }
        }
    }
    bn->n_numberNodes = 0;
    bn->numberNodes = NULL;

    return bn;
}

void freeNode(Node n){

    if (n->parents != NULL) free(n->parents);
    if (n->children != NULL) free(n->children);
    if (n->stateCountsTrue != NULL) free(n->stateCountsTrue);
    if (n->stateCountsFalse != NULL) free(n->stateCountsFalse);
    if (n->CPT != NULL) free(n->CPT);
    if (n->numberNodeChildren != NULL) free(n->numberNodeChildren);

    free(n);
}

void freeNumberNode(NumberNode nn){

    if (nn->parents != NULL) free(nn->parents);

    int n_parent_states = (int)(pow(2,nn->n_parents));
    if (nn->stateCounts != NULL){
        for ( int i = 0; i < n_parent_states; i++){
            free(nn->stateCounts[i]);
        }
        free(nn->stateCounts);
    }

    if (nn->CPT != NULL){
        for (int i = 0; i < n_parent_states; i++ ){
            free(nn->CPT[i]);
        }
        free(nn->CPT);
    }

    free(nn);
}

void freeBayesianNetwork(BayesianNetwork bn){
    int i,j,k;
    for (i = 0; i < bn->depth; i++){
        for (j = 0; j < bn->size; j++){
            for (k = 0; k < bn->size; k++){
                freeNode(bn->nodes[i][j][k]);
            }
            free(bn->nodes[i][j]);
        }
        free(bn->nodes[i]);
    }
    free(bn->nodes);
    if (bn->numberNodes != NULL){
        for(int i = 0; i < bn->n_numberNodes; i++){
            freeNumberNode(bn->numberNodes[i]);
        }
        free(bn->numberNodes);
    }
    free(bn);
}

int binaryToInt(bool* binaryNumber, int size){
    int result = 0;
    for (int i = 0; i < size; i++){
        result = result*2 + (binaryNumber[i] ? 1 : 0);
    }
    return result;
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

double probabilityGivenParentsNN(NumberNode nn){
    bool * parent_states = malloc(sizeof(bool) * nn->n_parents);
    double result;
    for (int i = 0; i < nn->n_parents; i++){
        parent_states[i] = nn->parents[i]->value;
    }
    result = nn->CPT[binaryToInt(parent_states,nn->n_parents)][nn->value];
    free(parent_states);
    return result;
}

int countsOfStateNumberNode(NumberNode nn){
    bool * parent_states = malloc(sizeof(bool) * nn->n_parents);
    for (int i = 0; i < nn->n_parents; i++){
        parent_states[i] = nn->parents[i]->value;
    }
    int res, row = binaryToInt(parent_states,nn->n_parents);
    res = 0;
    for (int i = 0; i < 10; i++){
        res += nn->stateCounts[row][i];
    }


    free(parent_states);
    return res;
}

void printBayesianNetwork(BayesianNetwork bn){
    printf("Bayesian network with depth: %d; size = %d; distanceRelations: %d and %d Number Nodes\n",bn->depth,bn->size,bn->distanceRelation, bn->n_numberNodes);
}

void printNode(Node n, bool printTables){
    printf("Node at position (d/x/y): %d %d %d\n",n->depth,n->x,n->y);
    Node n2;
    printf("parent Positions (d/x/y):\n");
    for (int i = 0; i < n->n_parents;i++){
        n2 = n->parents[i];
        printf("(%d %d %d) ",n2->depth,n2->x,n2->y);
    }
    printf("\n");
    printf("children Positions (d/x/y):\n");
    for (int i = 0; i < n->n_children;i++){
        n2 = n->children[i];
        printf("(%d %d %d) ",n2->depth,n2->x,n2->y);
    }
    printf("\n");
    int n_parent_configurations = (int)(pow(2,n->n_parents));
    if (printTables){
        printf("Counts (%d):\n",n_parent_configurations);
        if (n->stateCountsTrue == NULL|| n->stateCountsFalse == NULL){
            printf("NOT LEARNED\n");
        } else{
            for (int i = 0; i < n_parent_configurations ; i++){
                printf("ParentConfiguration %d: T: %d , F: %d\n",i,n->stateCountsTrue[i],n->stateCountsFalse[i]);
            }
        }

        printf("CBT(%d):\n",n_parent_configurations);
        if (n->CPT == NULL){
            printf("NOT LEARNED\n");
        } else{
            for (int i = 0; i < n_parent_configurations; i++){
                printf("ParentConfiguration %d: T: %.5f \n",i,n->CPT[i]);
            }
        }
    }
}

void printNumberNode(NumberNode nn, bool printTables){
    Node n2;
    printf("number node with value %d\n",nn->value);
    printf("%d parents at positions (d/x/y):\n", nn->n_parents);
    for (int i = 0; i < nn->n_parents;i++){
        n2 = nn->parents[i];
        printf("(%d %d %d) ",n2->depth,n2->x,n2->y);
    }
    printf("\n");
    int n_parent_configurations = (int)(pow(2,nn->n_parents));
    if (printTables){
        printf("Counts (%d):\n",n_parent_configurations);
        for (int i = 0; i < n_parent_configurations ; i++){
            printf("ParentConfiguration %d:",i);
            for(int j = 0; j < 10; j++){
                printf("\t%d", nn->stateCounts[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        

        printf("CBT(%d):\n",n_parent_configurations);
        for (int i = 0; i < n_parent_configurations ; i++){
            printf("ParentConfiguration %d:",i);
            for(int j = 0; j < 10; j++){
                printf("\t%f" , nn->CPT[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void setStateToData(BayesianNetwork bn, float *** data){
    for (int d = 0; d < bn->depth; d++){
        for (int x = 0; x < bn->size;x++){
            for (int y = 0; y < bn->size;y++){
                bn->nodes[d][x][y]->value = 0 < data[d][x][y] ;
            }
        }
    }

}



// with diagonals = true it also leanrs diagonal relations
void addAllDependencies(BayesianNetwork bn){

    int d,d2,x,y,x_relation,y_relation;
    Node n1,n2;
    int neighbourDistance = bn->distanceRelation;

    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (y = 0; y < bn->size; y++){
                n1 = bn->nodes[d][x][y];
                if (n1->parents == NULL){
                    n1->parents = malloc(sizeof(Node) * bn->depth * ( bn->diagonals ? 4:2));
                }
                if (n1->children == NULL){
                    n1->children = malloc(sizeof(Node)  * bn->depth * ( bn->diagonals ? 4:2));
                }
                n1->n_children = 0;
                n1->n_parents = 0;
            }
        }
    }

    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (y = 0; y < bn->size; y++){
                n1 = bn->nodes[d][x][y];
                for (x_relation = x-neighbourDistance; x_relation <= x+neighbourDistance; x_relation += neighbourDistance){
                    if (x_relation < 0 || x_relation >= bn->size){
                        continue;
                    }
                    for (y_relation = y-neighbourDistance; y_relation <= y+neighbourDistance; y_relation += neighbourDistance){
                        if (y_relation < 0 || y_relation >= bn->size){
                            continue;
                        }

                        if (y_relation < y || (y_relation == y && x_relation < x )){
                            
                            if (! bn->diagonals && !( y_relation == y || x_relation == x)){
                                //this is a diagonal relation
                                continue;
                            }

                            //n2 is a parents of n1 and  n1 is a child of n2
                            for  (d2 = 0; d2 < bn->depth; d2++){
                                n2 = bn->nodes[d2][x_relation][y_relation];
                                n1->parents[n1->n_parents] = n2;
                                n1->n_parents++;
                                n2->children[n2->n_children] = n1;
                                n2->n_children++;
                            }
                        }
                    }
                }
            }
        }
    }
}

//does not fir number nodes
void fitDataCountsOneLevel(BayesianNetwork bn, float **** data, int data_instances, int level){
    int i,d,x,y,j;
    Node n;

    //init counting arrays
    for (x = 0; x < bn->size; x++){
        for (y = 0; y < bn->size; y++){
            n = bn->nodes[level][x][y];

            if (n->stateCountsTrue != NULL) free(n->stateCountsTrue);
            if (n->stateCountsFalse != NULL) free(n->stateCountsFalse);

            n->stateCountsTrue = malloc(sizeof(int) * (int)(pow(2,n->n_parents)));
            n->stateCountsFalse = malloc(sizeof(int) * (int)(pow(2,n->n_parents)));   
            for (int i = 0; i < pow(2,n->n_parents); i++){
                n->stateCountsTrue[i] = 0;
                n->stateCountsFalse[i] = 0;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (x = 0; x < bn->size; x++){
        for (y = 0; y < bn->size; y++){

            Node n,parentNode;
            n = bn->nodes[level][x][y];
            bool *parentCombination = malloc(sizeof(bool) * n->n_parents);

            for (int i = 0; i < data_instances; i++){
                for (int j = 0; j < n->n_parents; j++ ){
                    parentNode = n->parents[j];
                    parentCombination[j] = 0 <data[i][parentNode->depth][parentNode->x][parentNode->y];
                }

                if (0 < data[i][level][x][y]){
                    n->stateCountsTrue[ binaryToInt(parentCombination,n->n_parents)]++;
                }else{
                    n->stateCountsFalse[ binaryToInt(parentCombination,n->n_parents)]++;
                }
            }
            free(parentCombination);
        }
    }
}

void fitDataCountsNumberNode(NumberNode nn, float **** data, int * nn_values, int n_data  ){

    int n_parent_combinations = pow(2,nn->n_parents);

    //reset state counts to zero
    for (int i = 0; i < n_parent_combinations; i++){
        for (int j = 0; j < 10; j++){
            nn->stateCounts[i][j] = 0;
        }
    }

    bool * parent_combination = malloc(sizeof(bool) * nn->n_parents);
    int row, parent_d, parent_x, parent_y;
    for (int i = 0; i < n_data; i++){
        for(int j = 0; j < nn->n_parents; j++){
            parent_d = nn->parents[j]->depth;
            parent_x = nn->parents[j]->x;
            parent_y = nn->parents[j]->y;
            parent_combination[j] = 0 < data[i][parent_d][parent_x][parent_y];
        }
        row = binaryToInt(parent_combination,nn->n_parents);
        nn->stateCounts[row][nn_values[i]]++;

    }

    free(parent_combination);
} 

void addDataCountsNumberNode(NumberNode nn, float **** data, int * nn_values, int n_data  ){

    int n_parent_combinations = pow(2,nn->n_parents);

    bool * parent_combination = malloc(sizeof(bool) * nn->n_parents);
    int row, parent_d, parent_x, parent_y;
    for (int i = 0; i < n_data; i++){
        for(int j = 0; j < nn->n_parents; j++){
            parent_d = nn->parents[j]->depth;
            parent_x = nn->parents[j]->x;
            parent_y = nn->parents[j]->y;
            parent_combination[j] = 0 < data[i][parent_d][parent_x][parent_y];
        }
        row = binaryToInt(parent_combination,nn->n_parents);
        nn->stateCounts[row][nn_values[i]]++;

    }

    free(parent_combination);
} 

//assumes parent relations are already known
//does not add for number nodes
void addDataCounts(BayesianNetwork bn, float **** data, int data_instances){
    int i,d,x,y,j;
    Node n;

    //init counting arrays, if necessary
    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (y = 0; y < bn->size; y++){
                n = bn->nodes[d][x][y];

                if (n->stateCountsTrue == NULL){
                    n->stateCountsTrue = malloc(sizeof(int) * (int)(pow(2,n->n_parents)));
                    for (int i = 0; i < pow(2,n->n_parents); i++){
                        n->stateCountsTrue[i] = 0;
                    }
                }
                if (n->stateCountsFalse == NULL){
                    n->stateCountsFalse = malloc(sizeof(int) * (int)(pow(2,n->n_parents)));   
                    for (int i = 0; i < pow(2,n->n_parents); i++){
                        n->stateCountsFalse[i] = 0;
                    }
                }
            }
        }
    }

    #pragma omp parallel for collapse(3)
    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (y = 0; y < bn->size; y++){

                Node n,parentNode;
                n = bn->nodes[d][x][y];
                bool *parentCombination = malloc(sizeof(bool) * n->n_parents);

                for (int i = 0; i < data_instances; i++){
                    for (int j = 0; j < n->n_parents; j++ ){
                        parentNode = n->parents[j];
                        parentCombination[j] = 0 < data[i][parentNode->depth][parentNode->x][parentNode->y];
                    }
                    if (0 < data[i][d][x][y]){
                        n->stateCountsTrue[ binaryToInt(parentCombination,n->n_parents)]++;
                    }else{
                        n->stateCountsFalse[ binaryToInt(parentCombination,n->n_parents)]++;
                    }
                    
                }
                free(parentCombination);
            }
        }
    }
}



// fit the data (in terms of counts), assumes that parents relations are already known
//does not fit numberNodes
void fitDataCounts(BayesianNetwork bn, float **** data, int n_data){

    int i,d,x,y,j;
    Node n;

    //init counting arrays
    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (y = 0; y < bn->size; y++){
                n = bn->nodes[d][x][y];

                if (n->stateCountsTrue != NULL) free(n->stateCountsTrue);
                if (n->stateCountsFalse != NULL) free(n->stateCountsFalse);

                n->stateCountsTrue = malloc(sizeof(int) * (int)(pow(2,n->n_parents)));
                n->stateCountsFalse = malloc(sizeof(int) * (int)(pow(2,n->n_parents)));   
                for (int i = 0; i < pow(2,n->n_parents); i++){
                    n->stateCountsTrue[i] = 0;
                    n->stateCountsFalse[i] = 0;
                }
            }
        }
    }

    #pragma omp parallel for collapse(3)
    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (y = 0; y < bn->size; y++){

                Node n,parentNode;
                n = bn->nodes[d][x][y];
                bool *parentCombination = malloc(sizeof(bool) * n->n_parents);

                for (int i = 0; i < n_data; i++){
                    for (int j = 0; j < n->n_parents; j++ ){
                        parentNode = n->parents[j];
                        parentCombination[j] = 0 < data[i][parentNode->depth][parentNode->x][parentNode->y];
                    }
                    if (0 < data[i][d][x][y]){
                        n->stateCountsTrue[ binaryToInt(parentCombination,n->n_parents)]++;
                    }else{
                        n->stateCountsFalse[ binaryToInt(parentCombination,n->n_parents)]++;
                    }
                }
                free(parentCombination);
            }
        }
    }
}

// determining the CPTs by using the counts
void fitCPTs(BayesianNetwork bn, float pseudoCounts){
    int d,x,y,p;
    Node n;

    #pragma omp parallel for private(d,x,y,p,n) collapse(3)
    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (y = 0; y < bn->size; y++){
                
                n = bn->nodes[d][x][y];
                if (n->CPT == NULL) n->CPT = malloc(sizeof(float) * pow(2,n->n_parents));

                for (p = 0; p < pow(2,n->n_parents); p++){
        
                    if ((float)(n->stateCountsTrue[p] + n->stateCountsFalse[p]) + pseudoCounts == 0){
                        n->CPT[p] = 0.5; //no data, no pseudo-counts
                    }else{
                        n->CPT[p] = ((float)(n->stateCountsTrue[p]) + pseudoCounts) / ( (float)(n->stateCountsTrue[p] + n->stateCountsFalse[p]) + 2*pseudoCounts);
                    }
                }
            }
        }
    }

    int n_parent_combinations, total_counts;
    NumberNode nn;
    #pragma omp parallel for private(nn,n_parent_combinations,total_counts,p)
    for (int i = 0; i < bn->n_numberNodes; i++){
        nn = bn->numberNodes[i];
        n_parent_combinations = (int)(pow(2,nn->n_parents));

        if (nn->CPT == NULL){
            nn->CPT = malloc(sizeof(float *) * n_parent_combinations);
            for (p = 0; p < n_parent_combinations; p++){
                nn->CPT[p] = malloc(sizeof(float) * 10);
            }

        }

        for (p = 0; p < n_parent_combinations; p++){
            total_counts = 0;
            for (int j = 0; j < 10; j++){
                total_counts += nn->stateCounts[p][j];
            }
            for (int j = 0; j < 10; j++){
                nn->CPT[p][j] = ((float)(nn->stateCounts[p][j] + pseudoCounts )) / ((float)( total_counts + 10 * pseudoCounts));
            }
        }

    }

}


//does not regard number nodes
float logMaxLikelihoodDataGivenModelOneLevel(BayesianNetwork bn, float **** data,int n_data,int level){
    fitDataCountsOneLevel(bn,data,n_data,level);
    
    int j,k,l;

    float logProb = 0;
    Node n;

    for ( j = 0; j < bn->size; j++){
        for ( k = 0; k < bn->size; k++){
            n = bn->nodes[level][j][k];
            for (l = 0; l < (int)(pow(2,n->n_parents));l++ ){
                if (n->stateCountsTrue[l]> 0){
                    logProb +=  (float)(n->stateCountsTrue[l]) * log((float)(n->stateCountsTrue[l]) / (n->stateCountsTrue[l] + n->stateCountsFalse[l]));
                }
                if (n->stateCountsFalse[l]> 0){
                    logProb +=  (float)(n->stateCountsFalse[l]) * log((float)(n->stateCountsFalse[l]) / (n->stateCountsTrue[l] + n->stateCountsFalse[l]));
                }
            }
        }
    }
    return logProb;

}

//calls fitDataCounts if updateCounts == true
//does not consider number nodes
float logMaxLikelihoodDataGivenModel(BayesianNetwork bn, float**** data, int n_instances, bool updateCounts){
    if (updateCounts){
        fitDataCounts(bn,data,n_instances);
    }

    int i,j,k,l;
    float logProb = 0;
    Node n;

    for ( i = 0; i < bn->depth; i++){
        for ( j = 0; j < bn->size; j++){
            for ( k = 0; k < bn->size; k++){
                n = bn->nodes[i][j][k];
                for (l = 0; l < (int)(pow(2,n->n_parents));l++ ){

                    if (n->stateCountsTrue[l]> 0){
                        logProb +=  n->stateCountsTrue[l] * log((float)(n->stateCountsTrue[l]) / (n->stateCountsTrue[l] + n->stateCountsFalse[l]));
                    }
                    if (n->stateCountsFalse[l]> 0){
                        logProb +=  n->stateCountsFalse[l] * log((float)(n->stateCountsFalse[l]) / (n->stateCountsTrue[l] + n->stateCountsFalse[l]));
                    }
                }
            }
        }
    }

    return logProb;
}

int numberParametersOneLevel(BayesianNetwork bn,int level){

    int n_parameters = 0;
    Node n;
    int j,k;

    for ( j = 0; j < bn->size; j++){
        for ( k = 0; k < bn->size; k++){
            n = bn->nodes[level][j][k];
            n_parameters += (int)(pow(2,n->n_parents));
        }
    }
    return n_parameters;
}

//does not consider numberNodes
int numberParameters(BayesianNetwork bn){
    int n_parameters = 0;
    Node n;

    int i,j,k;

    for ( i = 0; i < bn->depth; i++){
        for ( j = 0; j < bn->size; j++){
            for ( k = 0; k < bn->size; k++){
                n = bn->nodes[i][j][k];
                n_parameters += (int)(pow(2,n->n_parents));
            }
        }
    }
    return n_parameters;
}


//does not consider numberNodes
float bic(BayesianNetwork bn, float **** data, int n_data,  bool updateCounts, bool verbose){

    if (verbose){
        printf("n_parameters = %d, n_data = %d; log(n_data=  %f, logLikelihood =%f === %f\n"
            ,numberParameters(bn), n_data, log(n_data),logMaxLikelihoodDataGivenModel(bn,data,n_data, updateCounts)
            ,numberParameters(bn) * log(n_data) - 2 * logMaxLikelihoodDataGivenModel(bn,data,n_data, updateCounts));
    }

    return numberParameters(bn) * log(n_data) - 2 * logMaxLikelihoodDataGivenModel(bn,data,n_data, updateCounts);
}

//does not consider numberNodes
float bicOneLevel(BayesianNetwork bn, float **** data, int n_data, int level, bool verbose){

    if (verbose){
        printf("n_parameters = %d, n_data = %d; log(n_data) = %f; log_liklyhoodData = %f. ===> BIC %f\n",
        numberParametersOneLevel(bn,level) ,n_data, log(n_data), logMaxLikelihoodDataGivenModelOneLevel(bn,data,n_data, level),
         numberParametersOneLevel(bn,level) * log(n_data) - 2 * logMaxLikelihoodDataGivenModelOneLevel(bn,data,n_data, level));
    }

    return numberParametersOneLevel(bn,level) * log(n_data) - 2 
            * logMaxLikelihoodDataGivenModelOneLevel(bn,data,n_data, level);
}



