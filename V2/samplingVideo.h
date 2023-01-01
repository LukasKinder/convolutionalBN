

void generateImages(ConvolutionalBayesianNetwork cbn ,int * sequence, int len_sequence, int n_iterations, int n_images_per_instance, float smoothing){
    
    char name[30] = "./samples/sampleXXXX";
    bool *** samples;
    int c = 0;
    setToRandomState(cbn,0.7);
    for (int i  =0; i < len_sequence; i++){
        setValuesNumberNode(cbn,sequence[i]);
        samples = gibbsSampling(cbn,n_images_per_instance,n_iterations,smoothing);

        for (int j = 0; j < n_images_per_instance; j++){
            name[20] = (char)('0' + c %10);
            name[19] = (char)('0' + (c / 10) % 10);
            name[18] = (char)('0' + (c / 100) % 10);
            name[17] = (char)('0' + (c / 1000) % 10);

            scaleAndSaveImage(samples[j],28,name,15);
            c++;
        }

        freeImages(samples,n_images_per_instance,28);

    }

}