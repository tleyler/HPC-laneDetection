#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#define CHANNELS 3
#define BLOCK_SIZE 16
#define TILE_SIZE 16

__constant__ int gaussian[9];
__constant__ int sobel_x[9];
__constant__ int sobel_y[9];

__global__ void sobelKernel(unsigned char* deviceInput, unsigned char* deviceOutput, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int gx = 0; 
    int gy = 0;

    // no need to calculate sobel[3:5] because all those values will be 0
    gx = (sobel_x[0] * deviceInput[(y - 1) * width + (x - 1)]) +
         (sobel_x[1] * deviceInput[(  y  ) * width + (x - 1)]) +
         (sobel_x[2] * deviceInput[(y + 1) * width + (x - 1)]) +
         (sobel_x[6] * deviceInput[(y - 1) * width + (x + 1)]) +
         (sobel_x[7] * deviceInput[(  y  ) * width + (x + 1)]) +
         (sobel_x[8] * deviceInput[(y + 1) * width + (x + 1)]);

    gy = (sobel_y[0] * deviceInput[(y - 1) * width + (x - 1)]) +
         (sobel_y[1] * deviceInput[(y - 1) * width + (  x  )]) +
         (sobel_y[2] * deviceInput[(y - 1) * width + (x + 1)]) +
         (sobel_y[6] * deviceInput[(y + 1) * width + (x - 1)]) +
         (sobel_y[7] * deviceInput[(y + 1) * width + (  x  )]) +
         (sobel_y[8] * deviceInput[(y + 1) * width + (x + 1)]);

    deviceOutput[y * width + x] = sqrt((float)(gx * gx) + (gy * gy));
}

void sobelCuda(const cv::Mat& hostInput, cv::Mat& hostOutput)
{
    // Allocate memory on device for input and output
    unsigned char* deviceInput;
    unsigned char* deviceOutput;
    int bytes = hostInput.rows * hostInput.cols * sizeof(unsigned char);
    cudaMalloc((void**)&deviceInput, bytes);
    cudaMalloc((void**)&deviceOutput, bytes);

    // Copy memory from host to device 
    cudaMemcpy(deviceInput, hostInput.ptr(), bytes, cudaMemcpyHostToDevice);

    // Populate the global memory symbols with Sobel kernel values
    int h_sobel_x[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int h_sobel_y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    cudaMemcpyToSymbol(sobel_x, h_sobel_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(sobel_y, h_sobel_y, 9 * sizeof(int));

    // Call the kernel to convert the image to grayscale
    const dim3 numBlocks(ceil(hostInput.cols / BLOCK_SIZE), ceil(hostInput.rows / BLOCK_SIZE), 1);
    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    sobelKernel << <numBlocks, threadsPerBlock >> > (deviceInput, deviceOutput, hostInput.cols, hostInput.rows);

    // Copy memory back to host after kernel is complete
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput.ptr(), deviceOutput, bytes, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

__global__ void gaussianKernel(unsigned char* deviceInput, unsigned char* deviceOutput, int width, int height)
{
    const int KERNEL_SIZE = 9;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned char inputShared[TILE_SIZE];
    inputShared[threadIdx.x] = deviceInput[i];

    __syncthreads();

    int tileStartIdx =  blockIdx.x * blockDim.x;
    int tileEndIdx = (blockIdx.x + 1) * blockDim.x;

    int j = i - (KERNEL_SIZE / 2);

    int weight = 0;
    int sum = 0;

    for (int k = 0; k < KERNEL_SIZE; k++)
    {
        int current = j + k;
        if (current >= 0 && current < (width * height)) 
        {
            if (current >= tileStartIdx && current < tileEndIdx)
            {
                weight += inputShared[threadIdx.x + k - (KERNEL_SIZE / 2)] * gaussian[k];
                sum += gaussian[k];
            }
            else
            {
                weight += deviceInput[current] * gaussian[k];
                sum += gaussian[k];
            }
        }
    }
    deviceOutput[i] = static_cast<unsigned char>(weight / sum);
}


void gaussianCuda(const cv::Mat& hostInput, cv::Mat& hostOutput)
{
    // Allocate memory on device for input and output
    unsigned char* deviceInput;
    unsigned char* deviceOutput;
    int bytes = hostInput.rows * hostInput.cols * sizeof(unsigned char);
    cudaMalloc((void**) &deviceInput, bytes);
    cudaMalloc((void**) &deviceOutput, bytes);

    // Copy memory from host to device 
    cudaMemcpy(deviceInput, hostInput.ptr(), bytes, cudaMemcpyHostToDevice);

    // Populate the global memory symbol with Gaussian kernel values
    int hostGaussian[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    cudaMemcpyToSymbol(gaussian, hostGaussian, 9 * sizeof(int));

    // Call the kernel to convert the image to grayscale
    const dim3 gridSize((hostInput.rows * hostInput.cols + TILE_SIZE - 1) / TILE_SIZE, 1, 1);
    const dim3 blockSize(TILE_SIZE, 1, 1);
    gaussianKernel << < gridSize, blockSize >> > (deviceInput, deviceOutput, hostInput.cols, hostInput.rows); 

    // Copy memory back to host after kernel is complete
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput.ptr(), deviceOutput, bytes, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}


__global__ void grayscaleKernel(unsigned char* rgbInput, unsigned char* grayOutput, int width, int height, int colorWidthStep, int grayWidthStep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        const int colorOffset = y * colorWidthStep + (CHANNELS * x);
        const int grayOffset = y * grayWidthStep + x;

        const unsigned char b = rgbInput[colorOffset];
        const unsigned char g = rgbInput[colorOffset + 1];
        const unsigned char r = rgbInput[colorOffset + 2];

        const float gray = r * 0.3f + g * 0.59f + b * 0.11f;
        grayOutput[grayOffset] = static_cast<unsigned char>(gray);
    }
}

void grayscaleCuda(const cv::Mat& hostInput, cv::Mat& hostOutput)
{
    // Allocate memory on device for input and output
    unsigned char* deviceInput;
    unsigned char* deviceOutput;
    const int colorBytes = hostInput.step * hostInput.rows;
    const int grayBytes = hostOutput.step * hostOutput.rows;
    cudaMalloc<unsigned char>(&deviceInput, colorBytes);
    cudaMalloc<unsigned char>(&deviceOutput, grayBytes);

    // Copy memory from host to device
    cudaMemcpy(deviceInput, hostInput.ptr(), colorBytes, cudaMemcpyHostToDevice);

    // Call the kernel to convert the image to grayscale
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize((hostInput.cols + blockSize.x - 1) / blockSize.x, (hostInput.rows + blockSize.y - 1) / blockSize.y, 1); 
    grayscaleKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, hostInput.cols, hostInput.rows, hostInput.step, hostOutput.step);

    // Copy memory back to host after kernel is complete
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput.ptr(), deviceOutput, grayBytes, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

// This function takes in a path to a video file (which are passed in as command line args to main)
// as the first parameter and outputs each extracted frame to a vector of Mat items which is passed 
// in as the second parameter to the function. 
void extractFrames(const std::string& videoFilePath, std::vector<cv::Mat>& framesOut)
{
    try
    {
        cv::VideoCapture cap(videoFilePath);
        if (!cap.isOpened())
        {
            std::cerr << "Unable to open video file!" << std::endl;
            return;
        }
        for (int frameNum = 0; frameNum < cap.get(cv::CAP_PROP_FRAME_COUNT); frameNum++)
        {
            cv::Mat frame;
            cap >> frame;
            framesOut.push_back(frame);
            // VISUAL DEBUG: display each frame on screen
            // cv::imshow("Extracted Frame", frame);
            // waitKey(0);
        }
    }
    catch (cv::Exception& e)
    {
        std::cerr << e.msg << std::endl;
    }
}


// This function accepts a single frame and detects edges in it using opencv
// Canny(). It returns the edge detected image.
cv::Mat opencvCanny(const cv::Mat& frame) {
    // changing this number effects the amount of edges that it detects. The 
    // larger the number is, the less it will detect, only picking up larger 
    // edges. Through some quick experimentation I settled on 75, but this can
    // be adjusted later if we need more or less edges.
    double edgeThreshold = 75.0;

    // this mat will hold the edges image
    cv::Mat edgeDetectedFrame;
    cv::Canny(frame, edgeDetectedFrame, edgeThreshold, edgeThreshold * 3.0, 3);

    return edgeDetectedFrame;
}

// COMMAND LINE ARGUMENTS
// argv[0] = program name
// argv[1] = file path to video file
int main(int argc, char** argv)
{
    std::string videoFilePath = argv[1];
    std::vector<cv::Mat> framesOutput;
    extractFrames(videoFilePath, framesOutput);

    for (int i = 0; i < framesOutput.size(); i++)
    {
        cv::Mat image = framesOutput[i];
        const int rows = image.rows;
        const int cols = image.cols;
        //imshow("Extracted Frame", image);
        //cv::waitKey(0);

        // convert the image to grayscale 
        cv::Mat grayscale = cv::Mat(rows, cols, CV_8UC1);
        grayscaleCuda(image, grayscale);
        //imshow("Grayscale Image", grayscale);
        //cv::waitKey(0);

        // apply the Gaussian filter
        cv::Mat blurred = cv::Mat(rows, cols, CV_8UC1);
        gaussianCuda(grayscale, blurred);
        imshow("Blurred Image", blurred);
        cv::waitKey(0);

        // apply the Sobel operator
        cv::Mat sobel = cv::Mat(rows, cols, CV_8UC1);
        sobelCuda(blurred, sobel);
        imshow("Intensity Gradient Image", sobel);
        cv::waitKey();

        // Apply and output opencvCanny() to each extracted frame
        //imshow("Edge Detected Frame", opencvCanny(blurred));
        //cv::waitKey(0);
    }

    return 0;
}
