
#define NUMBER_NODE_DATA_PER_CONFIGURATION 100
#define NUMBER_NODE_PARENT_SUBSET_SIZE 0.01

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


//requires that the state counts are already known
double logMaxLikelihoodDataNumberNode(NumberNode nn){

    int n_rows = pow(2, nn->n_parents);
    int n_counts_row;
    double result = 0;
    for (int i = 0; i < n_rows; i++){
        n_counts_row = 0;
        for(int j = 0; j < 10; j++){
            n_counts_row += nn->stateCounts[i][j];
        }

        for(int j = 0; j< 10;j++){
            if (nn->stateCounts[i][j] != 0){
                result += (float)(nn->stateCounts[i][j]) * log( (float)(nn->stateCounts[i][j]) / (float)(n_counts_row));
            }
        }
    }
    return result;
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