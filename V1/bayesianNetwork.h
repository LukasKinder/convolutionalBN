
#include <math.h>

typedef struct RawNode *Node;

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

    int* stateCountsTrue; // maps parent configuration as index (binary number) to counts to be true
    int* stateCountsFalse; // maps parent configuration as index (binary number) to counts to be false
    float* CPT; // maps parent configuration as index (binary number) to probability to be true
  
}RawNode;

typedef struct RawBayesianNetwork *BayesianNetwork;

typedef struct RawBayesianNetwork {
    Node*** nodes; //nodes within the x,y,depth grid
    int size; //size in terms of x,y
    int depth; //size in terms of depth
    int distanceRelation;
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
    return n; 
}

void freeNode(Node n){
    free(n);
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

void setStateToData(BayesianNetwork bn, bool *** data){
    for (int d = 0; d < bn->depth; d++){
        for (int x = 0; x < bn->size;x++){
            for (int y = 0; y < bn->size;y++){
                bn->nodes[d][x][y]->value = data[d][x][y];
            }
        }
    }
}

BayesianNetwork createBayesianNetwork(int size, int depth){
    BayesianNetwork bn = malloc(sizeof(RawBayesianNetwork));
    bn->size = size;
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
    return bn;
}

// with diagonals = true it also leanrs diagonal relations
void addAllDependencies(BayesianNetwork bn, int neighbourDistance, bool diagonals){

    int d,d2,x,y,x_relation,y_relation;
    Node n1,n2;

    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (y = 0; y < bn->size; y++){
                n1 = bn->nodes[d][x][y];
                if (n1->parents == NULL){
                    n1->parents = malloc(sizeof(Node) * bn->depth * (diagonals ? 4:2));
                }
                if (n1->children == NULL){
                    n1->children = malloc(sizeof(Node)  * bn->depth * (diagonals ? 4:2));
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
                            
                            if (!diagonals && !( y_relation == y || x_relation == x)){
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

int binaryToInt(bool* binaryNumber, int size){
    int result = 0;
    for (int i = 0; i < size; i++){
        result = result*2 + (binaryNumber[i] ? 1 : 0);
    }
    return result;
}

//resets it to reverse any training, frees counts/CPT/parents
void resetBayesianNetwork(BayesianNetwork bn){
    int i,j,k;
    Node n;
    for(i = 0; i < bn->depth; i++){
        for (j = 0; j < bn->size; j++){
            for (k = 0; k < bn->size; k++){
                n = bn->nodes[i][j][k];
                if (n->parents != NULL) free(n->parents);
                if (n->children != NULL) free(n->children);
                if (n->stateCountsTrue != NULL) free(n->stateCountsTrue);
                if (n->stateCountsFalse != NULL) free(n->stateCountsFalse);
                if (n->CPT != NULL) free(n->CPT);

                n->parents = NULL;
                n->children = NULL;
                n->stateCountsTrue  = NULL;
                n->stateCountsFalse = NULL;
                n->CPT = NULL;
            }
        }
    }
}

void fitDataCountsOneLevel(BayesianNetwork bn, bool **** data, int data_instances, int level){
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
                    parentCombination[j] = data[i][parentNode->depth][parentNode->x][parentNode->y];
                }

                if (data[i][level][x][y]){
                    n->stateCountsTrue[ binaryToInt(parentCombination,n->n_parents)]++;
                }else{
                    n->stateCountsFalse[ binaryToInt(parentCombination,n->n_parents)]++;
                }
            }
            free(parentCombination);
        }
    }
}

// fit the data (in terms of counts), assumes that parents relations are already known
void fitDataCounts(BayesianNetwork bn, bool **** data, int data_instances){

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

                for (int i = 0; i < data_instances; i++){
                    for (int j = 0; j < n->n_parents; j++ ){
                        parentNode = n->parents[j];
                        parentCombination[j] = data[i][parentNode->depth][parentNode->x][parentNode->y];
                    }



                    if (data[i][d][x][y]){
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

// determining the CPTs by using the counts and laplaceCorrelation / smooting
void fitCPTs(BayesianNetwork bn, float pseudoCounts, bool cpt_smoothing, float** smoothing_kernel, int kernel_size){
    int d,x,y,p;
    Node n;

    #pragma omp parallel for private(d,x,y,p,n) collapse(3)
    for (d = 0; d < bn->depth; d++){
        for (x = 0; x < bn->size; x++){
            for (y = 0; y < bn->size; y++){
                
                n = bn->nodes[d][x][y];
                if (n->CPT == NULL) n->CPT = malloc(sizeof(float) * pow(2,n->n_parents));

                for (p = 0; p < pow(2,n->n_parents); p++){
                    if (cpt_smoothing){
                        printf("cpt smoothing not implemented\n");
                        exit(1);
                    }else{
                        
                        if ((float)(n->stateCountsTrue[p] + n->stateCountsFalse[p]) + pseudoCounts == 0){
                            n->CPT[p] = 0;
                        }else{
                            n->CPT[p] = ((float)(n->stateCountsTrue[p]) + pseudoCounts) / ( (float)(n->stateCountsTrue[p] + n->stateCountsFalse[p]) + 2*pseudoCounts);
                        }
                    }
                }
            }
        }
    }

}

void freeBayesianNetwork(BayesianNetwork bn){
    int i,j,k;
    resetBayesianNetwork(bn);
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
    free(bn);
}

float logMaxLikelihoodDataGivenModelOneLevel(BayesianNetwork bn,bool **** data,int n_data,int level){
    fitDataCountsOneLevel(bn,data,n_data,level);
    
    int j,k,l;

    float logProb = 0;
    Node n;

    for ( j = 0; j < bn->size; j++){
        for ( k = 0; k < bn->size; k++){
            n = bn->nodes[level][j][k];
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
    return logProb;

}

//calls fitDataCounts if updateCounts == true
float logMaxLikelihoodDataGivenModel(BayesianNetwork bn, bool**** data, int n_instances, bool updateCounts){

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


/*Todo approximate bic with fraction of parameters\n*/
float bic(BayesianNetwork bn, bool **** data, int n_data,  bool updateCounts){
    return numberParameters(bn) * log(n_data) - 2 * logMaxLikelihoodDataGivenModel(bn,data,n_data, updateCounts);
}

float bicOneLevel(BayesianNetwork bn, bool **** data, int n_data, int level){

    return numberParametersOneLevel(bn,level) * log(n_data) - 2 
            * logMaxLikelihoodDataGivenModelOneLevel(bn,data,n_data, level);
}



