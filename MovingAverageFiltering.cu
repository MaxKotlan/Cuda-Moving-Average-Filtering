#include <stdio.h>
#include <cuda.h>
#include <time.h>

struct Startup{
    int seed = time(nullptr);
    int random_range = 100;
    int threads_per_block = 1024;
    int data_size = 10000;
    int sample_size = 16;
    char* output_directory = ".";
    bool print = false;
    bool save = false;
    bool benchmark = false;
    bool single = false;
} startup;

/*
 Found on the stack overflow:  https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 Throws errors if cuda command doesn't return Success
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct DataSet{
    float* values;
    int  size;
};

inline int sizeOfDataSet(DataSet data){ return sizeof(float)*data.size; }

DataSet generateRandomDataSet(int size){
    DataSet data;
    data.size = size;
    data.values = (float*)malloc(sizeof(float)*data.size);

    for (int i = 0; i < data.size; i++)
        data.values[i] = (float)(rand()%startup.random_range);

    return data;
}

bool CompareDataSet(DataSet d1, DataSet d2){
    if (d1.size != d2.size) {printf("Datasets are not equal size\n"); return false;};

    for (int i = 0; i < d1.size; i++)
        if (d1.values[i] != d2.values[i]){
            printf("Dataset is different at %dth element. D1: %f, D2: %f", i, d1.values[i],  d2.values[i] );
            return false;
        }
    printf("D1 and D2 are equal!");
    return true;

}

/*A cache in-efficent algorithm for computing SMA. Loads everything form global memory*/
__global__ void DeviceCalculateSMA_Global(float* input, int input_size, float* result, int result_size, int sample_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < result_size){

        float sum = 0;
        for (int i = 0; i < sample_size; i++)
            sum += input[idx+i];
        sum /= sample_size;

        result[idx] = sum;
    }
}

/*A cache efficent algorithm for SMA. Each block loads range of data used by each of its threads into shared memory.
Then computes the moving average sum. This algorithm should becomes more efficent as the threads per block increases or
as the sample size increases
*/
__global__ void DeviceCalculateSMA_Shared(float* input, int input_size, float* result, int result_size, int sample_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < input_size){

        /*Shared memory. Size passed in with kernel parameters*/
        extern __shared__ float cache[];

        int cachedDataSize = sample_size + blockDim.x;

        /*Copy the data that will be used by the block into shared memory using all threads in the block.*/
        for (int i = 0; i < cachedDataSize/blockDim.x+1; i++){
            int cacheId = threadIdx.x+ i*blockDim.x;
            if (cacheId < cachedDataSize && cacheId+blockDim.x *blockIdx.x < input_size)
                cache[cacheId] = input[cacheId+blockDim.x *blockIdx.x];
        }
        __syncthreads();

        /*compute the sum using shared memory*/
        float sum = 0;
        for (int i = 0; i < sample_size; i++){
            if(i + threadIdx.x < cachedDataSize && i + idx < input_size)
                sum += cache[i+threadIdx.x];
        }
        sum /= sample_size;

        /*store in global memory*/
        if (idx < result_size)
            result[idx] = sum;
    }

}

DataSet CalculateSMA(DataSet input, int sample_size, bool usesharedmemory){
    if(sample_size == 1) { printf("Warning! Samplesize is 1. Result will equal input dataset.\n"); }
    if(input.size < 1) { printf("Cannot compute a moving average with an empty dataset.\n"); exit(-1); }
    if(sample_size < 1) { printf("Cannot compute a moving average with a samplesize of 0.\n"); exit(-1); }
    if(sample_size > input.size) { printf("Error! Sample Size is larger than dataset. Please make samplesize a value less than or equal to dataset size.\n"); exit(-1); }
    
    int result_size = input.size-sample_size+1;
    DataSet host_result = {(float*)malloc(sizeof(float)*(result_size)), result_size};

    float* device_input, *device_result;

    gpuErrchk(cudaMalloc((void **)&device_input,  sizeOfDataSet(input) ));
    gpuErrchk(cudaMalloc((void **)&device_result, sizeOfDataSet(host_result) ));

    gpuErrchk(cudaMemcpy(device_input, input.values, sizeOfDataSet(input) , cudaMemcpyHostToDevice));

    int threads_needed = host_result.size;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    if (usesharedmemory){
        int shared_memory_allocation_size = sizeof(float)*(startup.threads_per_block+sample_size);

        /*If shared memory too small, then optimized algorithm cannot be run. Exit*/
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (shared_memory_allocation_size > prop.sharedMemPerBlock) {printf("Cannot use shared Memory Algorithm. Not enough shared memory for dataset!"); exit(-1);}


        cudaEventRecord(start);
        DeviceCalculateSMA_Shared<<<threads_needed/ startup.threads_per_block + 1, startup.threads_per_block, shared_memory_allocation_size>>> (device_input, input.size, device_result, host_result.size, sample_size);
        cudaEventRecord(stop);

    }else{
        cudaEventRecord(start);
        DeviceCalculateSMA_Global<<<threads_needed/ startup.threads_per_block + 1, startup.threads_per_block>>> (device_input, input.size, device_result, host_result.size, sample_size);
        cudaEventRecord(stop);
    }

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (startup.single) {
        if (usesharedmemory) printf("Shared Memory: "); else printf("Global Memory: ");
        printf("Kernel executed in %f milliseconds\n", milliseconds);
    } else {
        if (usesharedmemory) printf("%.6g,", milliseconds);
        else printf("%.6g\n", milliseconds);
    }

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(host_result.values, device_result, sizeOfDataSet(host_result), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_result)); gpuErrchk(cudaFree(device_input));


    return host_result;
}

void printDataSet(DataSet data){
    for (int i = 0; i < data.size; i++)
        printf("%.6g, ", data.values[i]);
    printf("\n");
}

void saveDataSetCSV(DataSet data, char* fileName){

    char fileNameBuffer[256];

    snprintf(fileNameBuffer, sizeof fileNameBuffer, "%s/%s%s", startup.output_directory, fileName, ".csv");

    FILE* fp = fopen( fileNameBuffer, "w");
    if (fp == nullptr) printf("Could not log to file\n");
    else {
        for (int i = 0; i < data.size; i++){
            fprintf(fp, "%.6g,", data.values[i]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}


void AlgorithmsPerformanceBenchmark(){

    for (int i = 4; i <= 268435456; i*=2) {

        int j = i/2;
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        

        if (j > prop.sharedMemPerBlock/sizeof(float)) j = prop.sharedMemPerBlock/sizeof(float) - startup.threads_per_block;

        printf("%d,%d,", i, j);

        DataSet data = generateRandomDataSet(i);
        DataSet shared = CalculateSMA(data, j, true);
        DataSet global = CalculateSMA(data, j, false);

        free(data.values); free(shared.values); free(global.values);
    
    }
}


int main(int argc, char** argv){

    for (int i = 0; i < argc; i++){
        //if (strcmp(argv[i],  "--help")==0) {printf("%s", help); exit(-1); }
        if (strcmp(argv[i],  "--random_range")==0 && i+1 < argc) startup.random_range = atoi(argv[i+1]);
        if (strcmp(argv[i],  "--seed")==0 && i+1 < argc) startup.seed = atoi(argv[i+1]);
        if (strcmp(argv[i],  "--block_threads")==0 && i+1 < argc) startup.threads_per_block = atoi(argv[i+1]);


        if (strcmp(argv[i],  "--sample_size")==0 && i+1 < argc) startup.sample_size = atoi(argv[i+1]);
        if (strcmp(argv[i],  "--data_size")==0 && i+1 < argc) startup.data_size = atoi(argv[i+1]);


        if (strcmp(argv[i],  "--save")==0) startup.save = true;
        if (strcmp(argv[i],  "--print")==0) startup.print = true;
        if (strcmp(argv[i],  "--benchmark")==0) startup.benchmark = true;
        if (strcmp(argv[i],  "--single")==0) startup.single = true;

    }

    if (( startup.single || startup.benchmark ) == false)
        printf("Please select a runtime mode. There are two options --single or --benchmark\n\n\t--benchmark mode will continually increase the set size and sample size and compare the two algorithms.\n\n\t--single mode will apply SMA on a single randomly generated set. By default the dataset will be 10,000 elements with a sample size of 16. These parameters can be changes.\n\n");

    srand(startup.seed);

    if (startup.single) {

        DataSet data = generateRandomDataSet(startup.data_size);
        if(startup.print) printDataSet(data);
        if(startup.save)   saveDataSetCSV(data, "Input");

        DataSet shared = CalculateSMA(data, startup.sample_size, true);
        if(startup.print) printDataSet(shared);
        if(startup.save) saveDataSetCSV(shared, "Result");

        free(shared.values); free(data.values);
    }
    if (startup.benchmark)
        AlgorithmsPerformanceBenchmark();
}