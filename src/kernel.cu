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


// This function accepts a single frame and performs a hough transform on it 
// Returns a vector of the lines that were detected
void houghTransform(const cv::Mat& frame, std::vector<cv::Vec2f>& houghLines) {
    // create the houghLines vector with HoughLines method
    HoughLines(frame, houghLines, 1, CV_PI / 180, 150, 0, 0);
    return;
}

// This method determines the two best candidates out of all the lines picked
// up by the hough transform for the left and right lane, then draws them 
// on the original color frame image
cv::Mat drawLines(const cv::Mat& frame, std::vector<cv::Vec2f>& houghLines) {
    cv::Mat output = frame.clone();

    // handle edge case of not enough lines detected
    if (houghLines.size() < 2) {
        std::cerr << "Not enough lines detected with hough transform" << std::endl;
        return output;
    }

    // determine which lines will be the left and right lane lines
    int leftLaneCandidate = 0;
    float leftCandidateLeastDifference = 100.0f;
    int rightLaneCandidate = 0;
    float rightCandidateLeastDifference = 100.0f;
    float difference = 0.0f;

    // for the left lane you want to identify the line with theta greater than
    // 1/2 pi with the smallest difference of pi - theta
    // for the right lane you want to identify the line with theta less than 
    // 1/2 pi with the smallest theta
    for (size_t i = 0; i < houghLines.size(); i++)
    {
        float theta = houghLines[i][1];
        // looking at left lane candidate
        if (theta >= 1.57f) {
            difference = 3.14f - theta;
            // if true you have found a better left candidate
            if (difference < leftCandidateLeastDifference) {
                leftLaneCandidate = i;
                leftCandidateLeastDifference = difference;
            }
        }
        // looking at right lane candidate
        if (theta < 1.57f) {
            difference = theta;
            // if true you have found a better right candidate
            if (difference < rightCandidateLeastDifference) {
                rightLaneCandidate = i;
                rightCandidateLeastDifference = difference;
            }
        }
    }

    // seperate out just the identified lane lines
    std::vector<cv::Vec2f> lanes;
    lanes.push_back(houghLines[leftLaneCandidate]);
    lanes.push_back(houghLines[rightLaneCandidate]);


    // Draw the lines
    // Code for drawing lines on an image pulled from houghlines.cpp in opencv 
    // tutorials and adapted for our purpose
    for (size_t i = 0; i < lanes.size(); i++)
    {
        float rho = lanes[i][0], theta = lanes[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(output, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }
    return output;
}


cv::Mat gpuCanny(const cv::Mat &frame) {
    cv::Mat image = frame.clone();
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

    return image;
}

// COMMAND LINE ARGUMENTS
// argv[0] = program name
// argv[1] = file path to video file
int main(int argc, char** argv)
{
    std::string videoFilePath = argv[1];
    std::vector<cv::Mat> framesOutput;
    extractFrames(videoFilePath, framesOutput);

    bool gpuAccelerated = true;

    for (int i = 0; i < framesOutput.size(); i++)
    {
        cv::Mat edges;

        // This section is for when using the opencvCanny() implementation 
        // path (non-GPU accelarated)
        if (!gpuAccelerated) {
            // create Mat to hold the edges from canny edge detection
            edges = opencvCanny(framesOutput[i]);
            imshow("Edge Detected Frame", edges);
            cv::waitKey(0);
        }

        // This section is for when using our own GPU accelerated path
        else {
            // create Mat to hold the edges from canny edge detection
            edges = gpuCanny(framesOutput[i]);
            // imshow("Edge Detected Frame", edges);
            // cv::waitKey(0);
        }

        // currently houghlines only works when doing the non-gpu accelerated
        // path, but later it will be used on frame outputs from both, and this
        // if statement will be removed
        if (!gpuAccelerated) {
            // perform hough transform, storing lines detected in houghLines vector
            std::vector<cv::Vec2f> houghLines;
            houghTransform(edges, houghLines);
            imshow("lanes", drawLines(framesOutput[i], houghLines));
            cv::waitKey(0);
        }
    }

    return 0;
}
