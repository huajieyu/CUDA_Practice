#include <stdio.h>
#include <cuda.h>

__global__ void global_reduce_kernel(float *d_out, float *d_in, int n)
{
    //sdata is allocated in the kernel call: 3rd arg to <<b, t, shmem>
    extern __shared__ float sdata[];

    int myId = threadIdx + blockIdx.x * blockDim.x; //global Id
    int tid = threadIdx.x; //threadId

    //load shared mem from global mem
    s_data[tid] =d_in[myId];
    __syncthreads();  //make sure entire block is loaded!

    //do reduction in global memory
    for(unsigned int s= blockDim.x/2; s>0; s>>=1)
    {
        if(tid<2)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  //make sure all adds at one stage are done!
    }
    //only the 0th thread will be saved
    if(tid ==0)
    {
        d_out[blockIdx.x]=sdata[0];
    }
    
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 1024; // Number of elements
    const int BLOCK_SIZE = 256; // Number of threads per block
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float h_in[N], h_out[NUM_BLOCKS];

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f; // Example: set all values to 1.0
    }

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, NUM_BLOCKS * sizeof(float));
    
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    global_reduce_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_out, d_in, N);
    checkCUDAError("Kernel launch failed");

    // Synchronize and record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Copy the result back to the host
    cudaMemcpy(h_out, d_out, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate total reduction
    float total = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        total += h_out[i];
    }

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Output results
    printf("Total: %f\n", total);
    printf("Execution time: %f ms\n", milliseconds);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}