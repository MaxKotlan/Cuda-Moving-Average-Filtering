## Usage and Example Commands ###
This program consists of two modes.  *--benchmark* or *--single* You need to select one and only one of these modes.
### Single Mode

    MovingAverageFiltering.exe --single --print --data_size 100 --sample_size 16
This command will generate a data set with 100 elements, and will calculate a moving average with a sample period of 16. It will output the input and the result to the console.

    MovingAverageFiltering.exe --single --save --data_size 100000 --sample_size 100
    
This command will generate a data set with 100000 elements, and will calculate a moving average with a sample period of 100. It will output the input to a csv file named Input.csv and the result to a file named result.csv 

### Benchmark

Example Usage of benchmark Mode. Benchmark mode will output a csv file to the console (due to poor implementation on my part), that compares the performance of the global memory algorithm with the shared memory algorithm. 

    MovingAverageFiltering.exe --benchmark

The following is an example output from this command:

    4,2,0.00976,0.008672
    8,4,0.009728,0.00848
    16,8,0.009952,0.00848
    32,16,0.010336,0.008256
    64,32,0.010336,0.008672
    128,64,0.01056,0.009248
    256,128,0.011744,0.010144
    512,256,0.014848,0.013824
    1024,512,0.029408,0.026176
    2048,1024,0.049984,0.078784
    4096,2048,0.090944,0.15232
    8192,4096,0.17232,0.297536
    16384,8192,0.335584,0.86784
    32768,11264,0.898144,3.51942
    65536,11264,1.79443,8.32867
    131072,11264,3.59466,17.5548
    262144,11264,7.19786,35.7626
    524288,11264,14.4121,67.1892
    1048576,11264,26.044,134.082
    2097152,11264,51.6236,270.389
    4194304,11264,103.347,541.414
    8388608,11264,219.096,1087.15
    16777216,11264,423.523,2178.9
    33554432,11264,831.643,4361.3
    67108864,11264,1668.04,8718.71
    134217728,11264,3345.42,17453.5
    268435456,11264,6712,34884.2

This is in csv format (it's just output to the console instead of a file because of how I wrote the program). The left most column is the number of elements in the data set. The column to the right of that is the sample size. The column to the left of that is the shared memory algorithm's execution time (in milliseconds). The final column all the way to the right is the global memory algorithm's execution time (in milliseconds). This command was used for generating the graphs. 