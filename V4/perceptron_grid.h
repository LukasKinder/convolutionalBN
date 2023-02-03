
#define STEEPNESS_SIGMOID_PERCEPTRONS 1

#define INIT_NOISE_WEIGHTS 0.05

typedef struct RawNode *Node;

typedef struct RawNode {
    float value;
    float predictedValue;

    //position within grid
    int x;
    int y;
    int depth;

    float *** weights; // d/x/y 
    float bias;
  
}RawNode;


typedef struct RawPerceptronGrid *PerceptronGrid;

typedef struct RawPerceptronGrid {
    Node*** nodes; //nodes within the depth,x,y grid
    int size; //size in terms of x,y
    int depth; //size in terms of depth

    int maxDistanceRelations; //size of the max distance of nodes
    float distanceGaussianWidth;
    float distanceGaussianDistance;

    float* learning_curve;
    int learning_curve_size;
    int learning_curve_len;
}RawBayesianNetwork;

Node initNode(int depth, int x, int y, int max_distance_relations, int grid_depth){
    Node n = malloc(sizeof(RawNode));
    n->depth = depth;
    n->x = x;
    n->y = y;

    n->weights = malloc(sizeof(float **) * grid_depth);
    for (int d = 0; d < grid_depth; d++){
        n->weights[d] = malloc(sizeof(float *) * (2 * max_distance_relations +1));
        for (int x = 0; x < 2 * max_distance_relations +1; x++){
            n->weights[d][x] = malloc(sizeof(float) * (2 * max_distance_relations +1));
            for (int y = 0; y < 2 * max_distance_relations +1; y++){
                n->weights[d][x][y]  = INIT_NOISE_WEIGHTS *  ((float)(rand() - RAND_MAX/2) / (float)(RAND_MAX/2));
            }
        }
    }
    n->bias = 0.0;
    
    return n; 
}


PerceptronGrid createPerceptronGrid(int size, int depth, int max_distance_relations, float gaussian_width, float gaussian_distance){
    PerceptronGrid pg = malloc(sizeof(RawBayesianNetwork));
    pg->size = size;
    pg->depth = depth;

    pg->maxDistanceRelations = max_distance_relations;
    int i,j,k;


    pg->nodes = malloc(sizeof(Node**) * depth);
    for (i = 0; i < depth; i++){
        pg->nodes[i] = malloc(sizeof(Node*) * size);
        for (j = 0; j < size; j++){
            pg->nodes[i][j] = malloc(sizeof(Node) * size);
            for (k = 0; k < size; k++){
                pg->nodes[i][j][k] = initNode(i,j,k,max_distance_relations,depth);
            }
        }
    }


    pg->learning_curve = malloc(sizeof(float) * 1);
    pg->learning_curve_size = 1;
    pg->learning_curve_len = 0;
    return pg;
}


void freeNode(Node n, int max_distance_relations, int grid_depth){
    for (int d = 0; d < grid_depth; d++){
        for (int x = 0; x < 2 * max_distance_relations +1; x++){
            free(n->weights[d][x]);
        }
        free(n->weights[d]);
    }
    free(n->weights);


    free(n);
}

void freePerceptronGrid(PerceptronGrid pg){
    int i,j,k;
    for (i = 0; i < pg->depth; i++){
        for (j = 0; j < pg->size; j++){
            for (k = 0; k < pg->size; k++){
                freeNode(pg->nodes[i][j][k],pg->maxDistanceRelations,pg->depth);
            }
            free(pg->nodes[i][j]); 
        }
        free(pg->nodes[i]);
    }
    free(pg->nodes);


    free(pg->learning_curve);
    free(pg);
}

void saveLearningCurve(PerceptronGrid pg, char *name){
    char save_path[255] = "LearningCurves/";
    strncat(save_path,name, strlen(name) );

    printf("save path is: %s\n",name);
    FILE *fptr;
    fptr = fopen(save_path,"w");

    if(fptr == NULL){
        printf("Error when saving learning curve! cant open %s\n",save_path);   
        exit(1);             
    }

    for (int i = 0; i < pg->depth; i++){
        fprintf(fptr,"%f ", pg->learning_curve[i]);
        fprintf(fptr,"\n");
    }

    fclose(fptr);
}

double gaussian(float distance, float gaussian_distance, float gaussian_width){
    return exp( - pow(distance - gaussian_distance,2) / (2 * gaussian_width * gaussian_width));
}




void predict_node_value(Node n, PerceptronGrid pg){

    float result = 0.0;
    
    for (int d = 0; d < pg->depth; d++){
        for (int x = -pg->maxDistanceRelations; x < pg->maxDistanceRelations +1; x++){
            if (n->x + x > pg->size -1 || n->x + x < 0) continue;
            for (int y = -pg->maxDistanceRelations; y < pg->maxDistanceRelations +1; y++){
                if (n->y + y > pg->size-1 || n->y + y < 0) continue;

                result += pg->nodes[d][n->x + x][n->y +y]->value * n->weights[d][x + pg->maxDistanceRelations][y + pg->maxDistanceRelations];
            }
        }
    }
    n->predictedValue = sigmoid(result,STEEPNESS_SIGMOID_PERCEPTRONS);

}




void printNode(Node n, bool show_weights,int grid_depth, int max_distance_relations){
    printf("Node at position (d/x/y): %d %d %d\n",n->depth,n->x,n->y);
    printf("has value = %f, predicted value = %f\n",n->value,n->predictedValue);
    if (show_weights){
        
        for (int d = 0; d < grid_depth; d++){
            printf("d = %d:\n",d);
            for (int x = 0; x < 2 * max_distance_relations +1; x++){
                for (int y = 0; y < 2 * max_distance_relations +1; y++){
                    printf("%.3f ",n->weights[d][x][y]);
                }
                printf("\n");
            }
        }

        printf("bias = %f\n",n->bias);
    }
    
}


float familiarityState(PerceptronGrid pg){
    int i,j,k;
    float fam = 0.0;
    Node n;
    for ( i = 0; i < pg->depth; i++){
        for ( j = 0; j < pg->size; j++){
            for ( k = 0; k < pg->size; k++){
                n = pg->nodes[i][j][k];
                fam += pow(n->value - n->predictedValue,2);
            }
        }
    }
    return fam;
}

void setStateToData(PerceptronGrid pg, float *** data, bool fill_predicted_values){
    Node n;
    for (int  d = 0; d < pg->depth; d++){
        for (int x = 0; x < pg->size; x++){
            for (int y = 0; y < pg->size; y++){
                n = pg->nodes[d][x][y];
                n->value = data[d][x][y];
            }
        }
    }
    if (fill_predicted_values){
        for (int  d = 0; d < pg->depth; d++){
            for (int x = 0; x < pg->size; x++){
                for (int y = 0; y < pg->size; y++){
                    n = pg->nodes[d][x][y];
                    predict_node_value(n,pg);
                }
            }
        }           
    }
}