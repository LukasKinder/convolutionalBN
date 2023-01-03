

void initKernels(ConvolutionalBayesianNetwork cbn, int layer, float ***test_images, int n_test_images, bool verbose){

    int d = 1,s = 28;
    float ****layered_images, **** temp;
    layered_images = imagesToLayeredImagesContinuos(test_images,n_test_images,28);
    for (int l = 0; l < layer; l++){
        temp = layered_images;
        layered_images = dataTransition(temp,n_test_images,d,s,cbn->transitionalKernels[l],cbn->n_kernels[l],cbn->poolingKernels[l]);

        freeLayeredImagesContinuos(temp,n_test_images,d,s);

        d = cbn->bayesianNetworks[l]->depth;
        s = cbn->bayesianNetworks[l]->size;
    }

    if (verbose) printf("INIT_KERNELS: data init done\n");

    Kernel k,k_old;
    for (int i = 0; i < cbn->n_kernels[layer]; i++){
        k_old = cbn->transitionalKernels[layer][i];

        if (verbose) printf("INIT_KERNELS: create promising kernel %d of %d\n",i,cbn->n_kernels[layer]);

        cbn->transitionalKernels[layer][i] = createPromisingKernel(k_old.size,k_old.depth,k_old.stride,k_old.padding, layered_images,s,n_test_images,verbose);

        freeKernel(k_old);
    }
    
    freeLayeredImagesContinuos(layered_images,n_test_images,d,s);
    if (verbose) printf("INIT_KERNELS: Done\n");
}