#include <stdio.h>
#include <cuda.h>
#include <time.h>

struct Startup{
    int seed = time(nullptr);
    int random_range = 100;
    int threads_per_block = 1;
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

__global__ void DeviceCalculateSMA_Shared(float* input, int input_size, float* result, int result_size, int sample_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < result_size){

        extern __shared__ float cache[];

        int cachedDataSize = sample_size + blockDim.x;

        /*Copy the data that will be used by the block into shared memory using all threads in the block.*/
        for (int i = 0; i < cachedDataSize/blockDim.x; i++){
            int cacheId = threadIdx.x+ i*blockDim.x;
            if (cacheId < cachedDataSize)
                cache[cacheId] = input[cacheId+blockDim.x * blockIdx.x];
        }
        __syncthreads();

        float sum = 0;
        for (int i = 0; i < sample_size; i++){
            sum += cache[i+threadIdx.x];
        }

        sum /= sample_size;
        result[idx] = sum;
    }

}

DataSet CalculateSMA(DataSet input, int sample_size){
    int result_size = input.size-sample_size+1;
    DataSet host_result = {(float*)malloc(sizeof(float)*(result_size)), result_size};

    float* device_input, *device_result;

    gpuErrchk(cudaMalloc((void **)&device_input,  sizeOfDataSet(input) ));
    gpuErrchk(cudaMalloc((void **)&device_result, sizeOfDataSet(host_result) ));

    gpuErrchk(cudaMemcpy(device_input, input.values, sizeOfDataSet(input) , cudaMemcpyHostToDevice));

    int threads_needed = host_result.size;
    //DeviceCalculateSMA_Global<<<threads_needed/ startup.threads_per_block + 1, startup.threads_per_block>>> (device_input, input.size, device_result, host_result.size, sample_size);
    DeviceCalculateSMA_Shared<<<threads_needed/ startup.threads_per_block + 1, startup.threads_per_block, sizeof(float)*(input.size)>>> (device_input, input.size, device_result, host_result.size, sample_size);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(host_result.values, device_result, sizeOfDataSet(host_result), cudaMemcpyDeviceToHost));

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

    DataSet data = generateRandomDataSet(100);
    printDataSetI( data );
    DataSet result = CalculateSMA(data, 16);
    printf("\n");
    printDataSetF( result );
}