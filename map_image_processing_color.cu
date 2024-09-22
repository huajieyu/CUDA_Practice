//map is a key buiding block of a GPU computing
//1. parallel processors
//2. GPU is optimized for throughput rather then latency

//circulate the problems that can be solved using map
//1. sort an input array 
//2. add one to each element in an input array
//3. sum up all elements in an input array
//4. compute the average of an input array
//answer: 2 : all the calcualting to each element are parallep
//HW/project using the cuda to input a colored image and output that image in grey scall























__global__
void rgba_to_greyscale(const uchar4* const rgbaImage, 
                        unsigned char* const greyImage, 
                        int numRows, int numCols)
{
    //TODO
    //Fill in the kernel to convert from color to greyscale
    //the mapping from components of a uchar4 to RGBA is:
    // .x->R; .y -> G; .z ->B; .w ->A

    //The output(greyImage) at each pixel shoul dbe the result of
    //applying the formula : output = 0.2999f*R+.587*G+.114*B;
    //NOte we will be ingnoring the apha channel for this conversion

    //First createing a mapping from the 2D block and grid locations
    //to an absolute 2D location in the image, then use that to 
    //calculate a 1D offset
    // Calculate 2D index based on block and thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < numCols && y < numRows) {
        // Calculate the 1D offset for the current pixel
        // index 
        int index = y * numCols + x;

        // Fetch RGBA values from the input image
        uchar4 rgba = rgbaImage[index];
        unsigned char R = rgba.x;
        unsigned char G = rgba.y;
        unsigned char B = rgba.z;
        
        // Compute the greyscale value using the formula
        unsigned char grey = static_cast<unsigned char>(0.299f * R + 0.587f * G + 0.114f * B);

        // Store the greyscale value in the output image
        greyImage[index] = grey;
    }
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
    //you must fill in the correct sizes for the blockSIze and gridSize
    //current only one block with one thread is being launched
    //define block and grid sizes
    const int blockSize = 16;

    const dim3 blockDims(blockSize,blockSize,1); //TODO
    const dim3 gridDims((numCols + blockSize -1)/blockSize, (numRows + blockSize -1)/blockSize, 1); //TODO
    rgba_to_greyscale<<<gridDims, blockDims>>> (d_rgbaImage, d_greyImage, numRows, numCols);

    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}

int main(int argc, char** argv)
{
    // suppose you have image data
    const size_t numRows = 480;  // total rows
    const size_t numCols = 640;  // total columns

    // initialize and allocate memory
    uchar4* h_rgbaImage = new uchar4[numRows * numCols];
    unsigned char* h_greyImage = new unsigned char[numRows * numCols];

    // Fill up the data of h_rgbaImage
    // 
    for (size_t i = 0; i < numRows * numCols; ++i) {
        h_rgbaImage[i] = make_uchar4(255, 0, 0, 255);  // Pure Red Color
    }

    // allocate memory on Device
    uchar4* d_rgbaImage;
    unsigned char* d_greyImage;
    cudaMalloc(&d_rgbaImage, numRows * numCols * sizeof(uchar4));
    cudaMalloc(&d_greyImage, numRows * numCols * sizeof(unsigned char));

    // Copy the data from host to device 
    cudaMemcpy(d_rgbaImage, h_rgbaImage, numRows * numCols * sizeof(uchar4), cudaMemcpyHostToDevice);

    // call the function to convert rgba to greyscale
    your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows, numCols);

    // copy the result back to host from device
    cudaMemcpy(h_greyImage, d_greyImage, numRows * numCols * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // save the new image data
    // print out grey value
    for (size_t i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(h_greyImage[i]) << " ";
    }
    std::cout << std::endl;

    // free space
    delete[] h_rgbaImage;
    delete[] h_greyImage;
    cudaFree(d_rgbaImage);
    cudaFree(d_greyImage);

    return 0;
}