
typedef struct Node Node;

typedef struct Node {
    bool value;

    Node* parents;
    Node* children;
    int n_parents;
    int n_children;

    int* stateCountsTrue; // maps parent configuration as index (binary number) to counts to be true
    int* stateCountsFalse; // maps parent configuration as index (binary number) to counts to be false
    bool* CPT; // maps parent configuration as index (binary number) to probability to be true
  
}Node;

typedef struct BayesianNetwork {
    Node*** nodes; //nodes within the x,y,depth grid
    int size; //size in terms of x,y
    int depth; //size in terms of depth
}BayesianNetwork;

Node initNode(){
    Node n;
    n.parents = NULL;
    n.children = NULL;
    n.stateCountsTrue = NULL;
    n.stateCountsFalse = NULL;
    n.CPT = NULL;
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
                bn.nodes[i][j][k] = initNode();
            }

        }
    }
    return bn;
}


// fir the data by determining parents for each node and using counts for each node
void fitData(bool uniformParents, int neighbourDistance ){

}

// determining the CPTs by using the counts and laplaceCorrelation / smooting
void learnCPTs( bool cpt_smoothing, float** smoothing_kernel, int kernel_size, float pseudoCounts){


}

//resets it to reverse any training, frees counts/CPT/parents
void resetBayesianNetwork(BayesianNetwork bn){
    int i,j,k;
    Node n;
    for(i = 0; i < bn.depth; i++){
        for (j = 0; j < bn.size; j++){
            for (k = 0; k < bn.size; k++){
                n = bn.nodes[i][j][k];

                if (n.parents != NULL) free(n.parents);
                if (n.children != NULL) free(n.children);
                if (n.stateCountsTrue != NULL) free(n.stateCountsTrue);
                if (n.stateCountsFalse != NULL) free(n.stateCountsFalse);
                if (n.CPT != NULL) free(n.CPT);
            }
        }
    }
}

void freeBayesianNetwork(BayesianNetwork bn){
    int i,j;
    resetBayesianNetwork(bn);
    for (i = 0; i < bn.depth; i++){
        for (j = 0; j < bn.size; j++){
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