/*
Mainly taken by:
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C
*/

#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>



#define MNIST_LEN_INFO_IMAGE 4
#define MNIST_LEN_INFO_LABEL 2
#define MAX_FILENAME 256

void FlipLong(unsigned char * ptr)
{
    register unsigned char val;
    
    // Swap 1st and 4th bytes
    val = *(ptr);
    *(ptr) = *(ptr+3);
    *(ptr+3) = val;
    
    // Swap 2nd and 3rd bytes
    ptr += 1;
    val = *(ptr);
    *(ptr) = *(ptr+1);
    *(ptr+1) = val;
}

void image_char2bool(int num_data, unsigned char** data_image_char, bool*** data_image_bool, float threshold){
    int i, j, k;
    for (i=0; i<num_data; i++){
        for (j=0; j<28; j++){
            for (k=0; k<28; k++){
                data_image_bool[i][j][k]  = threshold  < (double)(data_image_char[i][j*28 + k] / 255.0 );
            }
        }
    }
}

bool*** readImages(char *file_path, int num_data, float threshold){

    int info_arr[MNIST_LEN_INFO_IMAGE];

    int i, j, k, fd;
    unsigned char *ptr;

    if ((fd = open(file_path, O_RDONLY)) == -1) {
        fprintf(stderr, "couldn't open image file");
        exit(-1);
    }

    
    read(fd, info_arr, MNIST_LEN_INFO_IMAGE * sizeof(int));

    unsigned char** data_char = malloc(sizeof(unsigned char*) * num_data);
    for (int i =0; i < num_data; i++){
        data_char[i] = malloc(sizeof(unsigned char) * 28*28);
    }

    // read-in mnist numbers (pixels|labels)
    for (i=0; i<num_data; i++) {
        read(fd, data_char[i], 28*28 * sizeof(unsigned char));   
    }

    //bool*** images[num_data][28][28];
    bool*** images = malloc(sizeof(bool**) * num_data);
    for(int i = 0; i < num_data; i++){
        images[i] = malloc(sizeof(bool*) *28);
        for(int j = 0; j < 28; j++){
            images[i][j] = malloc(sizeof(bool) *28);
        }
    }

    image_char2bool(num_data,data_char,images,threshold);

    close(fd);

    for (int i =0; i < num_data; i++){
        free(data_char[i]);
    }
    free(data_char);

    return images;
}

int * readLabels(char *file_path, int num_data){
    int info_arr[MNIST_LEN_INFO_LABEL];

    int i, j, k, fd;
    unsigned char *ptr;

    if ((fd = open(file_path, O_RDONLY)) == -1) {
        fprintf(stderr, "couldn't open image file");
        exit(-1);
    }

    
    read(fd, info_arr, MNIST_LEN_INFO_LABEL * sizeof(int));

    unsigned char** data_char = malloc(sizeof(unsigned char*) * num_data);
    for (int i =0; i < num_data; i++){
        data_char[i] = malloc(sizeof(unsigned char) * 1);
    }

    // read-in mnist numbers (pixels|labels)
    for (i=0; i<num_data; i++) {
        read(fd, data_char[i], 1 * sizeof(unsigned char));   
    }

    int * labels = malloc(sizeof(int) * num_data);
    for (int i  =0; i < num_data; i++){
        labels[i] = (int)(data_char[i][0]);
    }


    close(fd);
    for (int i =0; i < num_data; i++){
        free(data_char[i]);
    }
    free(data_char);

    return labels;
}

bool ** shiftImage(bool ** image, int size,int shift_right, int shift_up){
    bool ** shifted_image = malloc(sizeof(bool *) * size);
    for(int i = 0; i < size; i++){
        shifted_image[i] = malloc(sizeof(bool) * size);
        for(int j = 0; j < size; j++){
            if (i + shift_right < 0 || i + shift_right >= size || j + shift_up < 0 || j + shift_up >= size ){
                shifted_image[i][j] = false;
            }else {
                shifted_image[i][j] = image[i + shift_right][j + shift_up];
            }
        }
    }
    return shifted_image;
}

bool *** shiftImages(bool *** images,int n_data,int data_size,int shift_right, int shift_up){
    bool *** shifted_images = malloc(sizeof(bool**) * n_data);
    for(int i = 0; i < n_data; i++){
        shifted_images[i] = shiftImage(images[i],data_size,shift_right,shift_up);
    }
    return shifted_images;
}

bool **** imagesToLayeredImages(bool *** images, int n_data, int size){
    bool **** layeredImages;
    layeredImages = malloc(sizeof(bool ***) * n_data);
    for (int i =0; i < n_data; i++){
        layeredImages[i] = malloc(sizeof(bool**)*1);
        layeredImages[i][0] = malloc(sizeof(bool*) * size);
        for (int j = 0; j < size; j++){
            layeredImages[i][0][j] = malloc(sizeof(bool) * size);
            for (int k = 0; k < size; k++){
                layeredImages[i][0][j][k] = images[i][j][k];
                
            }
        }
    }
    return layeredImages;
}


void printImage(bool** image, int size){

    printf("    ");
    for (int i = 0; i < size; i++){
        if (i >= 10){
            printf("%d",i);
        }else {
            printf("%d ",i);
        }
    }
    printf("  \n");

    for (int x = 0; x < size; x ++){
        if (x >= 10){
            printf("%d",x);
        }else {
            printf("%d ",x);
        }
        printf("| ");
        for (int y =0; y < size; y ++){
            if (image[x][y]) {
                printf("X ");
            }else{
                printf("  ");
            }
        }
        printf(" |\n");
    }

    for (int i = 0; i < size +2; i++){
        printf("--");
    }
    printf("\n");
}


void freeImage(bool ** image, int size){
    for (int i = 0; i < size; i++){
        free(image[i]);
    }
    free(image);
}

void freeImages(bool*** images,int number, int size){
    for (int i = 0; i < number; i++){
        freeImage(images[i],size);
    }
    free(images);
}

void freeLayeredImages(bool**** layeredData, int data_size, int depth, int size){
    for (int i = 0; i < data_size; i++){
        freeImages(layeredData[i],depth,size);
    }

    free(layeredData);
}


void saveImage(bool ** image, int size, char name[]){
    char file_name[MAX_FILENAME];
    FILE *fp;
    int x, y;

    if (name[0] == '\0') {
        printf("output file name (*.pgm) : ");
        scanf("%s", file_name);
    } else strcpy(file_name, name);

    if ( (fp=fopen(file_name, "wb"))==NULL ) {
        printf("could not open file\n");
        exit(1);
    }

    fputs("P5\n", fp);
    fputs("# Created by Image Processing\n", fp);
    fprintf(fp, "%d %d\n", size, size);
    fprintf(fp, "%d\n", 255);
    for (y=0; y<size; y++)
        for (x=0; x<size; x++)
            fputc(image[y][x] ? 0: 255, fp);
    fclose(fp);
    //printf("Image was saved successfully\n");
}

void scaleAndSaveImage(bool ** image, int size, char name[], int scale){
    int new_size = size * scale;
    bool ** scaled_image = malloc(sizeof(bool) * size * new_size);
    for (int i = 0; i < new_size; i++){
        scaled_image[i] = malloc(sizeof(bool) * new_size);
        for (int j = 0; j < new_size; j++){
            scaled_image[i][j] = image[i / scale][j / scale];
        }
    }
    saveImage(scaled_image,new_size,name);

    freeImage(scaled_image,new_size);
}