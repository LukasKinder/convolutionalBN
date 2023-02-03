#define STRUCTURE_SEARCH_DATA_PER_CONFIGURATION 100


//the worst kernel is the one with the least outgoing relations
int determine_worst_kernel(BayesianNetwork bn, bool verbose){

    int  min_outgoing = 99999;
    int index_worst = -1;
    Node n;
    int current_outgoing;

    for (int i = 0; i < bn->depth; i++){

        n = bn->nodes[i][0][0];
        current_outgoing = n->n_children;

        //this does not include the children to the lower left diagonal, this needs to be corrected
        n = bn->nodes[i][bn->size -1 ][0];
        for (int j = 0; j < n->n_children; j++){
            if (n->children[j]->x < n->x){
                current_outgoing ++;
            }
        }

        if (verbose) printf("n outgoing depth %d is %d\n",i,current_outgoing);

        if (current_outgoing < min_outgoing || (current_outgoing == min_outgoing && rand() %5 == 0)){
            min_outgoing = current_outgoing;
            index_worst = i;
        }
    }

    if (verbose) printf("worst is kernel %d\n",index_worst);

    return index_worst;
}



void repeatedly_replace_worse_kernels(ConvolutionalBayesianNetwork cbn, int layer, float *** images, int n_data, int n_relations, int n_iterations, bool verbose){

    float ****temp, **** layered_images =  imagesToLayeredImagesContinuos(images, n_data, 28);
    int d = 1,s = 28;
    for (int l = 0; l < layer; l++){
        temp = layered_images;
        layered_images = dataTransition(temp,n_data,d,s
            ,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

        freeLayeredImagesContinuos(temp,n_data,d,s);

        d = cbn->bayesianNetworks[l]->depth;
        s = cbn->bayesianNetworks[l]->size;
    }

    BayesianNetwork bn = cbn->bayesianNetworks[layer];
    BayesianNetwork bn_before = cbn->bayesianNetworks[layer-1];

    if (n_relations == 0){
        printf("WARNING: 0 relations!");
        return;
    }

    int n_data_used = (int)(pow(2,n_relations)) * STRUCTURE_SEARCH_DATA_PER_CONFIGURATION ;
    if (n_data_used > n_data){
        printf("WARNING: not enough data for structure search (n_relations =n %d, n_data_per_row = %d, available data %d)\n",n_relations,STRUCTURE_SEARCH_DATA_PER_CONFIGURATION,n_data);
        n_data_used = n_data;
    }
    float **** data_next_layer;
    data_next_layer = dataTransition(layered_images, n_data_used,bn_before->depth,bn_before->size,cbn->transitionalKernels[layer]
                ,cbn->n_kernels[layer],cbn->poolingKernels[layer] );


    int index_worst_kernel;
    float *** structure_heuristics = initStructureHeuristic(bn->depth,n_relations,bn->diagonals);
    int n_updates;
    float ** tmp;
    int size_after;
    Kernel k,worst;

    int n_data_find_promising_kernel = 30;
    if (n_data_find_promising_kernel > n_data){
        n_data_find_promising_kernel = n_data;
    }

    //add child-parent-arrays to nodes with correct size

    for (int i = 0; i < n_iterations; i++){

        
        if (verbose) printf("Optimizing Structure\n");

        //printStructureHeuristic(structure_heuristics,bn->depth,n_relations,bn->diagonals);
        n_updates = updateStructureHeuristics(structure_heuristics,n_relations,bn,data_next_layer,n_data_used,STRUCTURE_SEARCH_DATA_PER_CONFIGURATION,false);
        if (verbose) printf("...required %d updates\n",n_updates);


        //printStructureHeuristic(structure_heuristics,bn->depth,n_relations,bn->diagonals);
        optimizeStructureUsingStructureHeuristic(bn,data_next_layer,n_data_used,structure_heuristics,n_relations,false);

        if (verbose) printf("Finding worst kernel\n");

        index_worst_kernel = determine_worst_kernel(bn, false);

        if (verbose) printf("REplacing worst kernel (%d)\n",index_worst_kernel);

        worst = cbn->transitionalKernels[layer][index_worst_kernel];
        k = createPromisingKernel(worst.size,worst.depth,worst.stride,worst.padding, layered_images,s,n_data_find_promising_kernel,false);
        cbn->transitionalKernels[layer][index_worst_kernel] = k;
        freeKernel(worst);

        if (verbose) printf("Update Structure Heuristic\n");

        updateStructureHeuristicNewKernel(structure_heuristics,bn->depth,n_relations,bn->diagonals,index_worst_kernel,false);

        if (verbose) printf("Update data\n");
        
        for (int j = 0; j < n_data_used; j++){
            freeImageContinuos(data_next_layer[j][index_worst_kernel],bn->size);

            tmp = applyConvolutionWeighted(layered_images[j],bn_before->size,bn_before->size,cbn->transitionalKernels[layer][index_worst_kernel],true);
            size_after = sizeAfterConvolution(bn_before->size,cbn->transitionalKernels[layer][index_worst_kernel]);
            data_next_layer[j][index_worst_kernel] = applyMaxPoolingOneLayer(tmp,size_after,size_after,cbn->poolingKernels[layer]);
            freeImageContinuos(tmp,size_after);
        }
        printf("Done updating Data\n");
    }
    freeStructureHeuristic(structure_heuristics,bn->depth,n_relations);
    freeLayeredImagesContinuos(data_next_layer,n_data_used,bn->depth,bn->size);
    d = cbn->bayesianNetworks[layer-1]->depth;
    s = cbn->bayesianNetworks[layer-1]->size;
    freeLayeredImagesContinuos(layered_images,n_data,d,s);
}