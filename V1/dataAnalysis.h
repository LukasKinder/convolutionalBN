



void printImageForEachKernel(bool ***layeredImage, int layer, int depth, int size){
    char name[25] = "rmageLayerXXKernelXX";
    for (int i  =0; i < depth; i++){
        name[10] = '0' + layer/10;
        name[11] = '0' + layer %10;
        name[18] = '0' + i / 10;
        name[19] = '0' + i %10;
        saveImage(layeredImage[i],size,name);
    }
}
