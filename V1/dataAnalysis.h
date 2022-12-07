



void printImageForEachKernel(bool ***layeredImage, int layer, int depth, int size){
    char name[25] = "imageKernelXXLayerXX";
    for (int i  =0; i < depth; i++){
        name[11] = '0' + i/10;
        name[12] = '0' + i %10;
        name[18] = '0' + layer / 10;
        name[19] = '0' + layer %10;
        saveImage(layeredImage[i],size,name);
    }
}
