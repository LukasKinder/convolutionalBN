
void printMeanImage(float** image, int size, float threshold){
    float mean_x = 0, mean_y = 0;
    int n_white = 0;

    for (int x = 0; x < size; x++){
        for (int y = 0; y < size; y++){
            if (image[x][y] > threshold){
                mean_x += x-14;
                mean_y += y -14;
                n_white +=1;
            }
        }
    }
    mean_x /=  n_white;
    mean_y /= n_white;
    printf("Mean pixel position at %.2f %.2f\n", mean_x,mean_y);
}
  

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


void rotateImageClockwise(float ** original, float ** target){
    for (int i = 0; i < 28; i++){
        for (int j = 0; j < 28; j++){
            target[27 - j][i] = original[i][j];
        }
    }
}

int canFindBestRotation(ConvolutionalBayesianNetwork cbn, float ** image, bool verbose){

    float ** temp;
    float ** copy = malloc(sizeof(float*) * 28);
    for (int i = 0; i < 28; i++){
        copy[i] = malloc(sizeof(float) * 28);
    }

    char name[255] = "imageXXXXXXXX";
    int r_id = rand();
    name[8] = (char)(r_id % 100000 /10000+ '0');
    name[9] = (char)(r_id % 10000 /1000+ '0');
    name[10] = (char)(r_id % 1000 /100+ '0');
    name[11] = (char)(r_id % 100 / 10 + '0');
    name[12] = (char)(r_id % 10 + '0');
    

    
    int best;
    float this, best_prob = -999999999999;
    for (int i = 0; i < 4;i++){
        setStateToImage(cbn,image);
        this = average_probability_nodes(cbn->bayesianNetworks[1]);
        //this = logLikelihoodState(cbn->bayesianNetworks[1]);
        
        if (verbose) printf("rotation %d, prob = %f\n",i,this);

        if (this >= best_prob){
            best = i;
            best_prob = this;
        }
        rotateImageClockwise(image,copy);
        temp = image;
        image = copy;
        copy = temp;

        
        /* printf("rotated %d: ",i); 
        printMeanImage(image,28,0.3);   */
        //saveImage(image,28,name,false);
        
    }

    name[13] = (char)('n');
    if (best == 0){
        name[14] = (char)('c');
    }else{
        name[14] = (char)('f');
    }
    //aveImage(image,28,name,false);

    freeImageContinuos(copy,28);
    return best;
}  

float rotatingImageAccuracy(ConvolutionalBayesianNetwork cbn, char* image_path, char * pathLabels, int n_data, bool verbose){

    if (verbose) printf("reading test data for rotating accuracy\n");
    float *** images = readImagesContinuos(image_path,n_data);
    int *labels = readLabels(pathLabels, n_data);

    /* char name[255] = "one_imageXXXXXXXX";
    for (int i = 0; i < n_data; i++){
        name[8] = (char)(i % 100000 /10000+ '0');
        name[9] = (char)(i % 10000 /1000+ '0');
        name[10] = (char)(i % 1000 /100+ '0');
        name[11] = (char)(i % 100 / 10 + '0');
        name[12] = (char)(i % 10 + '0');
        if (labels[i] == 1){
            saveImage(images[i],28,name,false);
        }
    }  */

    int *n_labels = calloc(10,sizeof(int));
    float **accuracy_n = malloc(sizeof(int *) * 10);
    for (int i = 0; i < 10; i++){
        accuracy_n[i] = calloc(4,sizeof(int));
    }

    if (verbose) printf("Done\n");

    float accuracy = 0;
    
    ConvolutionalBayesianNetwork * copys_cbn = malloc(sizeof(ConvolutionalBayesianNetwork) * N_THREADS);
    for (int i =0 ; i < N_THREADS; i++){
        copys_cbn[i] = copyConvolutionalBayesianNetwork(cbn);
    }

    int direction;

    #pragma omp parallel for num_threads(N_THREADS) private(direction)
    for (int i = 0; i < n_data; i++){
        if (verbose && i % 500 ==0) printf("reading image %d \t of %d\r",i, n_data);

        //if (labels[i]== 0) printMeanImage(images[i],28,0.3); 
        
 
        direction = canFindBestRotation(copys_cbn[omp_get_thread_num()],images[i],false);

        if (direction == 0){
            #pragma omp critical
            accuracy += 1.0;
        }
        
        #pragma omp critical
        {
        accuracy_n[labels[i]][direction] +=1;
        n_labels[labels[i]] +=1;
        }
    }

    for (int i =0 ; i < N_THREADS; i++){
        freeConvolutionalBayesianNetwork(copys_cbn[i]);
    }
    free(copys_cbn);
    


    accuracy /= (float)(n_data);

    if (verbose) printf("\nAccuracy each number: \n");
    for (int i = 0; i < 10; i++){
        if (verbose) printf("%d (%d): n: %f, o: %f, s: %f,w: %f,\n",i,n_labels[i],accuracy_n[i][0] / (float)(n_labels[i]),accuracy_n[i][1] / (float)(n_labels[i])
            ,accuracy_n[i][2] / (float)(n_labels[i]), accuracy_n[i][3] / (float)(n_labels[i]));
    }
    if (verbose) printf("\nDone\n ");

    for (int i = 0; i < 10; i++){
        free(accuracy_n[i]);
    }
    free(accuracy_n);
    free(n_labels);

    freeImagesContinuos(images,n_data,28);
    free(labels);

    return accuracy;
}