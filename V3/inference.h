



/* float *** gibbsSampling(ConvolutionalBayesianNetwork cbn, float ** startingImage, int n_samples, int iterations, float smoothing_factor){

    setStateToImage(cbn,startingImage);
    float ** currentImage = startingImage;

    float *** samples = malloc(sizeof(float**) * n_samples);
    int x,y;
    Node n;
    float prob_before, prob_after, p_change, p1,p2;
    float value_before;



    prob_before = probabilityState(cbn);

    samples[0] = copyImage(currentImage,28);

    for (int i = 1; i < n_samples; i++){
        printf("%d%\r",i);
        for (int j = 0; j < iterations / n_samples; j++){

            x = rand() % 28;
            y = rand() % 28;

            value_before = currentImage[x][y];
            currentImage[x][y] += 1 - 2 * ((float)(rand()) / RAND_MAX);

            if (currentImage[x][y] > 1) currentImage[x][y] = 1.0;
            if (currentImage[x][y] < 0) currentImage[x][y] = 0;

            printf("value before: %f, value after: %f\n",value_before, currentImage[x][y]);

            setStateToImage(cbn,currentImage);

            prob_after = probabilityState(cbn);
            printf("prob_before = %f, prob after = %f\n",prob_before,prob_after);
            p_change = prob_after / (prob_after + prob_before);

            p1 = pow(p_change,smoothing_factor);
            p2 = pow(1 - p_change, smoothing_factor);

            printf("intermediate: %f %f\n",p1,p2);

            p_change = p1 / (p1 + p2);

            printf("%f\n",p_change);

            if ((float)rand() / (float)RAND_MAX < p_change){
                prob_before = prob_after;
            }else{
                currentImage[x][y] = value_before;
            }
        }

        samples[i] = copyImage(currentImage,28);
    }
    return samples;
}
 */
//exponentially decreasing temperature, reached 0.005 at max iterations
float iterationToTemperature(float iteration, float maxIterations){
    return pow( pow(2.71828, -5.298317 / maxIterations) ,iteration);
}

float *** simulatedAnnealing(ConvolutionalBayesianNetwork cbn, int n_samples, float ** startingImage, int n_iterations){

    setStateToImage(cbn,startingImage);
    float ** currentImage = startingImage;

    float *** samples = malloc(sizeof(float**) * n_samples);
    int x,y;
    Node n;
    float prob_before, prob_after, p_change, p1,p2;
    float value_before;



    prob_before = probabilityState(cbn);

    samples[0] = copyImage(currentImage,28);

    for (int i = 1; i < n_samples; i++){
        printf("%d%\r",i);
        for (int j = 0; j < n_iterations / n_samples; j++){

            x = rand() % 28;
            y = rand() % 28;

            value_before = currentImage[x][y];
            currentImage[x][y] += 1 - 2 * ((float)(rand()) / RAND_MAX);

            if (currentImage[x][y] > 1) currentImage[x][y] = 1.0;
            if (currentImage[x][y] < 0) currentImage[x][y] = 0;

            setStateToImage(cbn,currentImage);

            prob_after = probabilityState(cbn);
            printf("prob_before = %f, prob after = %f\n",prob_before,prob_after);
            p_change = prob_after / (prob_after + prob_before);



            p_change = p1 / (p1 + p2);

            printf("%f\n",p_change);

            if ( p_change > 0.5 || (float)rand() / (float)RAND_MAX < 0.01){
                prob_before = prob_after;
            }else{
                currentImage[x][y] = value_before;
            }
        }

        samples[i] = copyImage(currentImage,28);
    }
    return samples;
}

float *** strictClimbing(ConvolutionalBayesianNetwork cbn, int n_samples, float ** startingImage, int n_iterations){

    setStateToImage(cbn,startingImage);
    float ** currentImage = startingImage;

    float *** samples = malloc(sizeof(float**) * n_samples);
    int x,y;
    Node n;
    float prob_before, prob_after, p_change;
    float value_before;



    prob_before = probabilityState(cbn);

    samples[0] = copyImage(currentImage,28);

    for (int i = 1; i < n_samples; i++){
        printf("%d%\r",i);
        for (int j = 0; j < n_iterations / n_samples; j++){

            x = rand() % 28;
            y = rand() % 28;

            value_before = currentImage[x][y];
            currentImage[x][y] += 1 - 2 * ((float)(rand()) / RAND_MAX);

            if (currentImage[x][y] > 1) currentImage[x][y] = 1.0;
            if (currentImage[x][y] < 0) currentImage[x][y] = 0;

            setStateToImage(cbn,currentImage);

            prob_after = probabilityState(cbn);
            printf("prob_before = %f, prob after = %f\n",prob_before,prob_after);
            p_change = prob_after / (prob_after + prob_before);

            printf("%f\n",p_change);

            if ( p_change > 0.5 ){
                prob_before = prob_after;
            }else{
                currentImage[x][y] = value_before;
            }
        }

        samples[i] = copyImage(currentImage,28);
    }
    return samples;
}

float obstructionTask(ConvolutionalBayesianNetwork cbn, float ** startingImage, int n_iterations, int obstructionSize){

    saveImage(startingImage,28,"original",false);

    float ** original = malloc(sizeof(float*) * obstructionSize);
    int x_obstruction = rand() % (28 - obstructionSize);
    int y_obstruction = rand() % (28 - obstructionSize);
    for (int i = 0; i < obstructionSize; i++){
        original[i] = malloc(sizeof(float) * obstructionSize);
        for (int j = 0; j < obstructionSize; j++){
            original[i][j] = startingImage[x_obstruction + i][y_obstruction+j];
            startingImage[x_obstruction + i][y_obstruction+j] = (float)(rand()) / RAND_MAX;
        }
    }

    saveImage(startingImage,28,"randomized",false);

    setStateToImage(cbn,startingImage);
    float ** currentImage = startingImage;
    int x,y;
    float prob_before, prob_after, p_change;
    float value_before;



    prob_before = probabilityState(cbn);

    for (int j = 0; j < n_iterations; j++){

        x = x_obstruction + rand() % obstructionSize;
        y = y_obstruction + rand() % obstructionSize;

        value_before = currentImage[x][y];
        currentImage[x][y] = 1.0 - 2 * ((float)(rand()) / RAND_MAX);

        if (currentImage[x][y] > 1.0) currentImage[x][y] = 1.0;
        if (currentImage[x][y] < 0.0) currentImage[x][y] = 0.0;

        setStateToImage(cbn,currentImage);

        prob_after = probabilityState(cbn);
        p_change = prob_after / (prob_after + prob_before);

        printf("%f ->",p_change);

        if ( p_change >= 0.5 ){
            printf("(accept)\n");
            prob_before = prob_after;
        }else{
            printf("(reject)\n");
            currentImage[x][y] = value_before;
        }
    }
    
    saveImage(startingImage,28,"reconstructed",false);

    float euclideanDistance = 0.0;

    for (int i = 0; i < obstructionSize; i++){
        for (int j = 0; j < obstructionSize; j++){
            if (currentImage[x_obstruction + i][y_obstruction+j] < 0.2) currentImage[x_obstruction + i][y_obstruction+j] = 0.0;

            euclideanDistance += pow(original[i][j] - currentImage[x_obstruction + i][y_obstruction+j],2);
        }
    }
    

    for (int i = 0; i < obstructionSize; i++){
        free(original[i]);
    }
    free(original);

    return sqrt(euclideanDistance);
}


/* void saveBestWorst(bool *** images, int n_images, int n_best ,ConvolutionalBayesianNetwork cbn){
    float * image_probs = malloc(sizeof(float) * n_images);
    for(int i = 0; i < n_images; i++){
        setStateToImage(cbn,images[i]);
        image_probs[i] = logProbabilityStateCBN(cbn); //TODO UPDATE, in order to implement depenent
    }

    //save the best,worst

    char name_best[10] = "bestX";
    char name_worst[10] = "worstX";
    int best_index,worst_index;
    float current_best,current_worst, last_best = 0,last_worst = -999999999;
    for (int i  =0; i < n_best; i++ ){
        current_best = -999999999;
        current_worst = 0;
        for( int j = 0; j < n_images; j++){
            if (image_probs[j] > current_best && image_probs[j] < last_best){
                current_best = image_probs[j];
                best_index = j;
            }
            if (image_probs[j] < current_worst && image_probs[j] > last_worst){
                current_worst = image_probs[j];
                worst_index = j;
            }
        }
        last_best = current_best;
        last_worst = current_worst;


        name_best[4] = '0' + i;
        saveImage(images[best_index],28,name_best);

        name_worst[5] = '0' + i;
        saveImage(images[worst_index],28,name_worst);
        

    }
    free(image_probs);
} */