
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
    bool* CPT; // maps parent configuration as index (binary number) to probability to be true
  
}RawNode;

typedef struct BayesianNetwork {
    Node*** nodes; //nodes within the x,y,depth grid
    int size; //size in terms of x,y
    int depth; //size in terms of depth
}BayesianNetwork;

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

BayesianNetwork createBayesianNetwork(int size, int depth){
    BayesianNetwork bn;
    bn.size = size;
    int i,j,k;
    bn.depth = depth;
    bn.nodes = malloc(sizeof(Node**) * depth);
    for (i = 0; i < depth; i++){
        bn.nodes[i] = malloc(sizeof(Node*) * size);
        for (j = 0; j < size; j++){
            bn.nodes[i][j] = malloc(sizeof(Node) * size);
            for (k = 0; k < size; k++){
                bn.nodes[i][j][k] = initNode(i,j,k);
            }
        }
    }
    return bn;
}

//
void addAllDependencies(BayesianNetwork bn){
    int d,d2,x,y,x_relation,y_relation;
    Node n1,n2;

    for (d = 0; d < bn.depth; d++){
        for (x = 0; x < bn.size; x++){
            for (y = 0; y < bn.size; y++){
                n1 = bn.nodes[d][x][y];
                if (n1->parents == NULL){
                    n1->parents = malloc(sizeof(Node) * 4 * bn.depth);
                }
                if (n1->children == NULL){
                    n1->children = malloc(sizeof(Node) * 4 * bn.depth);
                }
                n1->n_children = 0;
                n1->n_parents = 0;
            }
        }
    }

    for (d = 0; d < bn.depth; d++){
        for (x = 0; x < bn.size; x++){
            for (y = 0; y < bn.size; y++){
                n1 = bn.nodes[d][x][y];
                for (x_relation = x-1; x_relation <= x+1; x_relation++){
                    if (x_relation < bn.size || x_relation >= bn.size){
                        continue;
                    }
                    for (y_relation = y-1; y_relation <= y+1; y_relation++){
                        if (y_relation < bn.size || y_relation >= bn.size){
                            continue;
                        }

                        if (y_relation < y || (y_relation == 0 && x_relation < x )){
                            //n2 is a parents of n1 and  n1 is a child of n2
                            for  (d2 = 0; d2 < bn.depth; d2++){
                                n2 = bn.nodes[d2][x_relation][y_relation];
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

// fit the data (in terms of counts), assumes that parents relations are already known
void fitDataCounts(BayesianNetwork bn, bool **** data, int data_instances){

    int i,d,x,y,j;
    Node n;

    printf("A\n");

    //init counting arrays
    for (d = 0; d < bn.depth; d++){
        for (x = 0; x < bn.size; x++){
            for (y = 0; y < bn.size; y++){
                n = bn.nodes[d][x][y];
                n->stateCountsTrue = malloc(sizeof(int) * (int)(pow(2,n->n_parents)));
                n->stateCountsFalse = malloc(sizeof(int) * (int)(pow(2,n->n_parents)));   
                for (int i = 0; i < pow(2,n->n_parents); i++){
                    n->stateCountsTrue[i] = 0;
                    n->stateCountsFalse[i] = 0;
                }
            }
        }
    }

    printf("B\n");

    #pragma omp parallel for collapse(3)
    for (d = 0; d < bn.depth; d++){
        for (x = 0; x < bn.size; x++){
            for (y = 0; y < bn.size; y++){

                Node n,parentNode;
                bool *parentCombination = malloc(sizeof(bool) * 4 * bn.depth);
                n = bn.nodes[d][x][y];

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
    printf("C\n");

}

// determining the CPTs by using the counts and laplaceCorrelation / smooting
void learnCPTs(float pseudoCounts, bool cpt_smoothing, float** smoothing_kernel, int kernel_size){


}

//resets it to reverse any training, frees counts/CPT/parents
void resetBayesianNetwork(BayesianNetwork bn){
    int i,j,k;
    Node n;
    for(i = 0; i < bn.depth; i++){
        for (j = 0; j < bn.size; j++){
            for (k = 0; k < bn.size; k++){
                n = bn.nodes[i][j][k];
                if (n->parents != NULL) free(n->parents);
                if (n->children != NULL) free(n->children);
                if (n->stateCountsTrue != NULL) free(n->stateCountsTrue);
                if (n->stateCountsFalse != NULL) free(n->stateCountsFalse);
                if (n->CPT != NULL) free(n->CPT);
            }
        }
    }
}

void freeBayesianNetwork(BayesianNetwork bn){
    int i,j,k;
    resetBayesianNetwork(bn);
    for (i = 0; i < bn.depth; i++){
        for (j = 0; j < bn.size; j++){
            for (k = 0; k < bn.size; k++){
                free(bn.nodes[i][j][k]);
            }
            free(bn.nodes[i][j]);
        }
        free(bn.nodes[i]);
    }
    free(bn.nodes);
}

void numberParameters(BayesianNetwork bn){

}

// Uses only counts
float maxLikelihood(BayesianNetwork bn){

}

// returns the log likelihood of the current state
float logLikelihoodState(BayesianNetwork bn){

}

//sets the state of each node to the data
float setToState(BayesianNetwork bn, bool*** data){
    
}



/*
1) init kernel such that resulting data of kernel has low correlation with data of other kernels at same node,
BUT has a high infomation gain for data of neighboring nodes.
2) Structure learning from scratch => Goal is to maximize maximum-likelihood of data.
3) update kernel through search => Goal is to maximize "maximum-likelihood of data with relations" - "maximum likelihood of data without relations".
4) repeat 2) and 3) until convergence, for 2) use result of previous 2) as starting point (or random otherwise). 



BUT: how to prevent too similar kernels during repetition of 2) and 3) ? -> maybe not a problem?
*/

/*
Easy solution:
Search through kernels such that kernels have low correlation to each other on same position but high information gain 
for neighbour positions.

Do stucture/parameter search afterwards 

*/

/*
Bad solution:
Do not use structure search. Fully connected. Find kernel that maximize likelihood without outgoing vs with outgoing. 
*/

/*
Use handmade heuristic:

Goodness of bn is: "correlation-at-same-position" + "likelihood-of-data" + "number parameters"



*/