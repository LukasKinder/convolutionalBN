
void predictionNumberOneLayer(ConvolutionalBayesianNetwork cbn, int layer,  float * prob_dist){
    BayesianNetwork bn = cbn->bayesianNetworks[layer];
    NumberNode nn;
    for (int i = 0; i < 10; i++){
        prob_dist[i] = 0;
    }

    //! assumes that all number nodes have the same amount of parents
    bool * parent_states = malloc(sizeof(bool) * bn->numberNodes[0]->n_parents);

    for (int i = 0; i < bn->n_numberNodes; i++){
        nn = bn->numberNodes[i];

        for (int i = 0; i < nn->n_parents; i++){
            parent_states[i] = nn->parents[i]->value;
        }


        for (int j = 0; j < 10; j++){
            prob_dist[j] +=  nn->CPT[binaryToInt(parent_states,nn->n_parents)][j];
        }
    }
    free(parent_states);
}

void predictNumber(ConvolutionalBayesianNetwork cbn, int *prediction1, int * prediction2, bool verbose ){
    float * overall_prob_dist = calloc(sizeof(float), 10);
    float * local_prob_dist = malloc(sizeof(float) * 10);

    int n_layer_with_nn = 0;
    for (int i = 0; i < cbn->n_layers; i++){

        if (cbn->bayesianNetworks[i]->n_numberNodes == 0){
            continue;
        }
        n_layer_with_nn++;


        predictionNumberOneLayer(cbn, i,local_prob_dist);

        if (verbose){
            printf("Probability distribution based on layer %d \n", i);
            for (int j = 0; j < 10; j++){
                printf("%f.3 \t", local_prob_dist[j]);
            }
            printf("\n");
        }
        for (int j = 0; j < 10; j++){
            overall_prob_dist[j] += local_prob_dist[j];
        }
    }

    int n_number_nodes = 0;
    for (int i = 0; i < cbn->n_layers; i++){
        n_number_nodes += cbn->bayesianNetworks[i]->n_numberNodes;
    }

    for (int j = 0; j < 10; j++){
        overall_prob_dist[j] /= (float)(n_number_nodes);
    }

    if (verbose){
        printf(" overall probability distribution:\n");
        for (int j = 0; j < 10; j++){
            printf("%f.3 \t", overall_prob_dist[j]);
        }
        printf("\n");
    }

    float best_prob = -1;
    for (int j = 0; j < 10; j++){
        if (overall_prob_dist[j] > best_prob){
            best_prob = overall_prob_dist[j];
            *prediction1 = j;
        }
    }

    best_prob = -1;
    for (int j = 0; j < 10; j++){
        if (overall_prob_dist[j] > best_prob && j != *prediction1){
            best_prob = overall_prob_dist[j];
            *prediction2 = j;
        }
    }

    free(overall_prob_dist);
    free(local_prob_dist);
}

/* void predictNumberManualExperiment(ConvolutionalBayesianNetwork cbn, float *** images, int n_images, int * labels){

    int random_image, X;

    for (int i = 0; i < 30; i++){
        printf("predict another random image? (<0 = no else yes)\n");
        scanf("%d", &X);
        if (! X) break;

        random_image = rand() % n_images;
        setStateToImage(cbn,images[random_image]);
        printImageContinuos(images[random_image],28);
        printf("is %d, predicts %d\n", labels[random_image], predictNumber(cbn,true));

    }
} */

float predictNumberAccuracy(ConvolutionalBayesianNetwork cbn, char* image_path, char * label_path, int n_data, bool verbose){

    if (verbose) printf("reading test data\n");
    int * labels = readLabels(label_path, n_data);
    float *** images = readImagesContinuos(image_path,n_data);

    if (verbose) printf("Done\n");

    float accuracy = 0;
    float lenient_accuracy = 0;
    int guess1, guess2;
    for (int i = 0; i < n_data; i++){
        if (verbose && i % 50 ==0) printf("reading image %d \t of %d\r",i, n_data);
        setStateToImage(cbn, images[i]);
        predictNumber(cbn,&guess1, &guess2,false);
        if (guess1== labels[i] ){
            accuracy +=1;
        }
        if (guess1 == labels[i] || guess2 == labels[i]){
            lenient_accuracy +=1;
        }
    }
    if (verbose) printf("\nDone\n ");

    freeImagesContinuos(images,n_data,28);
    free(labels);

    printf("in best 2: %f\n", lenient_accuracy / n_data);

    return accuracy / n_data;
}