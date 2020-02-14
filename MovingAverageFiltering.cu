#include <stdio.h>
#include <cuda.h>
#include <time.h>

struct Startup{
    int seed = time(nullptr);
    int random_range = 10;
} startup;

struct DataSet{
    int* values;
    int  size;
};

DataSet generateRandomDataSet(int size){
    DataSet data;
    data.size = size;
    data.values = (int*)malloc(sizeof(int)*data.size);

    for (int i = 0; i < data.size; i++)
        data.values[i] = rand()%startup.random_range;

    return data;
}

void printDataSet(DataSet data){
    for (int i = 0; i < data.size; i++)
        printf("%d ", data.values[i]);
    printf("\n");
}

int main(int argc, char** argv){
    printDataSet( generateRandomDataSet(100) );
}