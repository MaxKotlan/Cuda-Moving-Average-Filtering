#include <stdio.h>
#include <cuda.h>
#include <time.h>

struct Startup{
    int seed = time(nullptr);
    int random_range = 100;
    int threads_per_block = 256;
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

        extern __shared__ float cache[];

        int cachedDataSize = sample_size + blockDim.x;

        /*Copy the data that will be used by the block into shared memory using all threads in the block.*/
        for (int i = 0; i < cachedDataSize/blockDim.x+1; i++){
            int cacheId = threadIdx.x+ i*blockDim.x;
            if (cacheId < cachedDataSize && cacheId+blockDim.x *blockIdx.x < input_size)
                cache[cacheId] = input[cacheId+blockDim.x *blockIdx.x];
        }
        __syncthreads();

        float sum = 0;
        for (int i = 0; i < sample_size; i++){
            if(i + threadIdx.x < cachedDataSize && i + idx < input_size)
                sum += cache[i+threadIdx.x];
        }

        sum /= sample_size;
        result[idx] = sum;
    }

}

void printTime(clock_t totaltime){
    int msec = totaltime / CLOCKS_PER_SEC;
    printf("Done in %d micro sec!\n", msec);
}

DataSet CalculateSMA(DataSet input, int sample_size, bool usesharedmemory){
    int result_size = input.size-sample_size+1;
    DataSet host_result = {(float*)malloc(sizeof(float)*(result_size)), result_size};

    float* device_input, *device_result, *cacheDebug;

    gpuErrchk(cudaMalloc((void **)&device_input,  sizeOfDataSet(input) ));
    gpuErrchk(cudaMalloc((void **)&device_result, sizeOfDataSet(host_result) ));
    gpuErrchk(cudaMalloc((void **)&cacheDebug, sizeof(float)*startup.threads_per_block+sample_size))

    gpuErrchk(cudaMemcpy(device_input, input.values, sizeOfDataSet(input) , cudaMemcpyHostToDevice));

    int threads_needed = host_result.size;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    if (usesharedmemory){
        int shared_memory_allocation_size = sizeof(float)*(startup.threads_per_block+sample_size);
        cudaEventRecord(start);
        DeviceCalculateSMA_Shared<<<threads_needed/ startup.threads_per_block + 1, startup.threads_per_block, shared_memory_allocation_size>>> (device_input, input.size, device_result, host_result.size, sample_size);
        cudaEventRecord(stop);

    }else{
        cudaEventRecord(start);
        DeviceCalculateSMA_Global<<<threads_needed/ startup.threads_per_block + 1, startup.threads_per_block>>> (device_input, input.size, device_result, host_result.size, sample_size);
        cudaEventRecord(stop);
    }
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(host_result.values, device_result, sizeOfDataSet(host_result), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if (usesharedmemory) printf("Shared Memory: ");
    else printf("Global Memory: ");
    printf("Kernel executed in %f milliseconds\n", milliseconds);

    return host_result;
}

void printDataSetI(DataSet data){
    for (int i = 0; i < data.size; i++)
        printf("%.0f,", data.values[i]);
    printf("\n");
}

void printDataSetF(DataSet data){
    for (int i = 0; i < data.size; i++)
        printf("%.4f ", data.values[i]);
    printf("\n");
}


int main(int argc, char** argv){
    srand(0);

    DataSet data = generateRandomDataSet(10000);
    //printDataSetI( data );
    DataSet shared = CalculateSMA(data, 1325, true);
    DataSet global = CalculateSMA(data, 1325, false);

    //printf("\n");
    //printDataSetF( shared );
    //printf("\n");
    //printDataSetF( global );
    //printf("\n");


    printf("Each should be %d elements in size\n", global.size);
    CompareDataSet(global, shared);
}