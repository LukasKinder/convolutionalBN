
#define PRETRAINING_ITERATION_PER_PARAMETER  1 //suggested: 10
#define PRETRAINING_N_DATA 50 //suggested: 100
#define PRETRAINING_N_INIT_IMAGES 10 //suggested 10

//TODO: run with openmp

float covariance(bool **** data, int n_data, int kernel1_depth,int kernel2_depth, int size, float mean1, float mean2){
    float covariance = 0;
    int i,j,k;
    for ( i  =0; i < n_data; i++){
        for (j = 0; j < size; j++){
            for (k = 0; k < size; k++){
                covariance += ((data[i][kernel1_depth][j][k] ? 1.0 :0.0) - mean1)
                            * ((data[i][kernel2_depth][j][k] ? 1.0 :0.0) - mean2);  
            }
        }
    }
    return covariance / (n_data * size * size);
}

//Todo: consider including diagonal relations
float covarianceNeighbors(bool **** data,int n_data, int kernel_depth, int n_kernels,int size, float * means, int neighbour_distance){
    float covariance = 0;
    int i,j,k,d,x,y;
    int c = 0;
    for ( i  =0; i < n_data; i++){
        for (j = 0; j < size; j++){
            for (k = 0; k < size; k++){
                for (d = 0; d < n_kernels; d++){
                    //only upper and left neighbour
                    if (j - neighbour_distance >=0){
                        covariance += ((data[i][kernel_depth][j][k] ? 1.0 :0.0) - means[kernel_depth])
                                    * ((data[i][d][j-neighbour_distance][k] ? 1.0 :0.0) - means[d]);  
                        c++;
                    }
                    if (k - neighbour_distance >=0){
                        covariance += ((data[i][kernel_depth][j][k] ? 1.0 :0.0) - means[kernel_depth])
                                    * ((data[i][d][j][k - neighbour_distance] ? 1.0 :0.0) - means[d]);  
                        c++;
                    }
                }
            }
        }
    }

    return covariance / c;
}

float meanKernelValues(bool **** data, int n_data, int kernel_depth, int size){
    float mean = 0;
    int i,j,k;
    for ( i  =0; i < n_data; i++){
        for (j = 0; j < size; j++){
            for (k = 0; k < size; k++){
                mean += data[i][kernel_depth][j][k] ? 1.0 :0.0; 
            }
        }
    }
    return mean / (n_data * size * size);
}


float varianceKernelValues(bool **** data, int n_data, int kernel_depth, int size, float mean){

    int n_1 = (int)( (n_data *size * size) * mean);
    int n_0 = (n_data *size * size) -n_1;
    return ((1.0 - mean)*(1.0 - mean) * n_1 + (0.0 - mean)*(0.0 - mean)* n_0) / (n_data *size * size);
}

float correlation(bool **** data, int n_data, int kernel1_depth,int kernel2_depth, int size, float mean1, float mean2){

    float variance1 = varianceKernelValues(data,n_data,kernel1_depth,size,mean1);
    float variance2 = varianceKernelValues(data,n_data,kernel2_depth,size,mean2);

    if (variance1 == 0 || variance2 == 0) return 1.0; //Edge-case

    return covariance(data,n_data,kernel1_depth,kernel2_depth,size,mean1,mean2) 
            / (sqrt(variance1) * sqrt(variance2));
}

//sum Kernels  FOREACH KERNEL: (1 - |correlation()|) (sum correlation(Neighbours) + a * correlation(digit) )
//Todo add correlation with digit
float pretrainingHeuristic(bool **** data, int n_data, int depth, int size, int neighbour_distance){
    float heuristic = 0;
    float * means = malloc(sizeof(float) * depth);
    int n_zero_or_one = 0;
    for (int i = 0; i < depth; i++){
        means[i] = meanKernelValues(data,n_data,i,size);
        if (means[i] == 0 || means[i] == 1) n_zero_or_one++;
    }
    if (n_zero_or_one != 0){
        free(means);
        return (float)(n_zero_or_one) * (-1); // default heuristic, smaller if many zero/one means
    }

    float highestCorrelation, current_correlation;
    for (int i = 0; i < depth; i++){
        highestCorrelation = 0;
        for (int j = 0; j < depth; j++){
            if (i == j){
                continue;
            }
            current_correlation = fabs( correlation(data,n_data,i,j,size,means[i],means[j]) );

            if (current_correlation > highestCorrelation) highestCorrelation = current_correlation;

        }

        heuristic += (1 - highestCorrelation) 
            * ( fabs(covarianceNeighbors(data,n_data,i,depth,size,means,neighbour_distance)) );
    }

    free(means);
    return heuristic;
}

//inits kernel until it is not all black or all white in first three images
void initKernelsToMeaningful(Kernel * kernels, int n_kernels,bool ****data_previous, int n_data, int size_data , bool verbose){
    int n_testing = n_data < PRETRAINING_ITERATION_PER_PARAMETER ? n_data : PRETRAINING_ITERATION_PER_PARAMETER;

    bool all_meaningful = false;
    bool **** newData;
    int new_size = size_data;
    new_size = sizeAfterConvolution(new_size,kernels[0]);
    Kernel non_pooling_kernel = createKernel(1,n_kernels,pooling,1,false); //does not need to be freed because it is a pooling kernel

    float mean;
    Kernel old_kernel;
    int it = 0;

    if (verbose) printf("making sure that default kernels are all meaningful\n");

    while (!all_meaningful){
        if (it == 10000){
            if (verbose) printf("Failed to find meaningful kernels\n",it);
            break;
        }

        if (verbose) printf("Searching for meaningful kernels (iteration %d)\n",it);

        all_meaningful = true;
        newData = dataTransition(data_previous,n_testing,kernels[0].depth,size_data,kernels,n_kernels,non_pooling_kernel );
        for (int i = 0; i < n_kernels; i++){
            mean =  meanKernelValues(newData,n_testing,i,new_size);
            if (mean < 0.0001 || mean > 0.05){
                //too rare, not meaningful
                old_kernel = kernels[i];
                kernels[i] = createKernel(old_kernel.size,old_kernel.depth,old_kernel.type,old_kernel.stride,old_kernel.padding);
                freeKernel(old_kernel);
                all_meaningful = false;
            }
        }
        freeLayeredImages(newData,n_testing,n_kernels,new_size);
        it++;
    }
    if (verbose) printf("Found meaningful kernels\n");

}

void pretrainKernels(Kernel * kernels, int n_kernels, Kernel poolingKernel
        , bool ****data_previous, int n_data, int size_data,int  neighbour_distance, bool verbose ){
    if (n_kernels <=1){
        printf("Error: pretraining is not designed to be used for just one kernel\n");
        exit(0);
    }

    if (verbose) printf("Pretrain Kernels\n");

    initKernelsToMeaningful(kernels,n_kernels,data_previous,n_data,size_data,verbose);

    return;

    //number of iterations is 10 times the amount of parameters
    int n_iterations = (int) (PRETRAINING_ITERATION_PER_PARAMETER 
            * n_kernels * kernels[0].depth * kernels[0].size * kernels[0].size);

    Kernel randomKernel;
    int randD,randX,randY, index_rand_kernel;
    MT_MF_E_Value previous;

    bool **** newData;

    int depth_previous = kernels[0].depth;
    int size_kernels = kernels[0].size;
    int usedData = n_data < PRETRAINING_N_DATA  ? n_data : PRETRAINING_N_DATA; 

    int size_after_normal_kernels = sizeAfterConvolution(size_data,kernels[0]);
    int size_after = sizeAfterConvolution(size_after_normal_kernels, poolingKernel);

    newData = dataTransition(data_previous,usedData,depth_previous,size_data,kernels,n_kernels,poolingKernel);

    float new_heuristic, current_heuristic;
    current_heuristic =  pretrainingHeuristic(newData,usedData,n_kernels,size_after,neighbour_distance);

    if (verbose) printf("PRETRAINING: initialization done\n");

    for (int iteration = 0; iteration < n_iterations; iteration++){
        index_rand_kernel = rand() % n_kernels;
        randomKernel = kernels[index_rand_kernel];
        randD = rand() % depth_previous;
        randX = rand() % size_kernels;
        randY = rand() % size_kernels;
        previous = randomKernel.map[randD][randX][randY];
        while (randomKernel.map[randD][randX][randY] == previous) {
            switch (rand() % 3){
            case 0:
                randomKernel.map[randD][randX][randY] = must_true;
                break;
            case 1:
                randomKernel.map[randD][randX][randY] = must_false;
                break;
            case 2:
                randomKernel.map[randD][randX][randY] = either;
                break;
            }
        }


        #pragma omp parallel for
        for(int i =0; i < usedData; i++){
            freeImage(newData[i][index_rand_kernel],size_after);
            bool ** intermediate_result  = applyConvolution(data_previous[i],size_data,size_data,randomKernel);
            newData[i][index_rand_kernel] = applyMaxPoolingOneLayer(intermediate_result,size_after_normal_kernels,size_after_normal_kernels,poolingKernel);
            freeImage(intermediate_result,size_after_normal_kernels);
        }
        
        new_heuristic = pretrainingHeuristic(newData,usedData,n_kernels,size_after,neighbour_distance);

        if (verbose   && iteration % 30 == 0 ) printf("Iteration %d / %d current heuristic: %f, new_heuristic: %f",iteration,n_iterations,current_heuristic ,new_heuristic);

        if (new_heuristic > current_heuristic){
            //keep change
            current_heuristic = new_heuristic;
            if (verbose   && iteration % 30 == 0 ) printf("  ===> accepted\n");
        }else{
            //discard Change
            randomKernel.map[randD][randX][randY] = previous;
            if (verbose   && iteration % 30 == 0 ) printf("  ===> rejected\n");
        }
    }

    freeLayeredImages(newData,usedData,n_kernels,size_after);

    if (verbose){
        printf("Done\n Kernels are:\n");
        for (int i = 0; i < n_kernels; i++){
            printKernel(kernels[i]);
        }
    }
}