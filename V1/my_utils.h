
#include <math.h>

//stores number as "number * 10^(power)"
typedef struct SuperSmall {
    float number;
    int power;
  
}SuperSmall;

SuperSmall initSuperSmall(float value){
    SuperSmall n;
    n.number = value;
    n.power = 0;
}

float floatOfSuperSmall(SuperSmall n){
    return n.number * pow(10.0, (float)(n.power));
}

void standardizeSuperSmall(SuperSmall *n){
    while (n->number > 10){
        n->number /= 10;
        n->power++;
    }
    while (n->number < 1){
        n->number *= 10;
        n->power--;
    } 
}

void multiplySuperSmallF(SuperSmall * n1, float n2){
    if (n2 == 0.0){
        printf("Error: SuperSmall can not represent 0!\n");
        exit(1);
    }
    n1->number *=n2;
    standardizeSuperSmall(n1);
}

SuperSmall multiplySuperSmalls(SuperSmall n1, SuperSmall n2){
    SuperSmall result;
    result.power = n1.power + n2.power;
    result.number = n1.number + n2.number;
    standardizeSuperSmall(&result);
    return result;
}

float divideSuperSmalls(SuperSmall n1, SuperSmall n2){
    while (n1.power < 1 || n2.power <1){
        n1.power++;
        n2.power++;
    }
    return floatOfSuperSmall(n1) / floatOfSuperSmall(n2);
}

void printSuperSmall(SuperSmall n){
    printf("%.2f*10^%d",n.number,n.power);
}

bool isBiggerSuperSmall(SuperSmall n1, SuperSmall n2){
    return n1.power > n2.power || ( n1.power == n2.power && n1.number > n2.number);
}

SuperSmall addSupersmalls(SuperSmall n1, SuperSmall n2){
    SuperSmall smaller;
    SuperSmall bigger;
    if (isBiggerSuperSmall(n1,n2)){
        bigger = n1; smaller =n2;
    }else{
        bigger = n2; smaller =n1;
    }

    if (bigger.power - smaller.power > 6){
        //no point in adding
        return bigger;
    }

    while (smaller.power != bigger.power){
        bigger.number *= 10;
        bigger.power--;

    }
    smaller.number += bigger.number;
    standardizeSuperSmall(&smaller);
    return smaller;
}