#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#define PI 3.14159265
#define CHANNELS 3
#define BLOCK_SIZE 16
#define TILE_SIZE 16
#define STRONG_EDGE 255
#define WEAK_EDGE 125
#define NON_EDGE 0
#define HYST_LOW 75
#define HYST_HIGH 120

__constant__ int gaussian[9];
__constant__ int sobel_x[9];
__constant__ int sobel_y[9];



// This method is the kernel for the final step of hysteresis thresholding
__global__ void hysteresisKernel(unsigned char* deviceInput, unsigned char* 
    deviceOutput, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char intensity = deviceInput[y * width + x];

    if (intensity == WEAK_EDGE)
    {
        // check all the neighbors for strong edges
        bool strongNeighbor = false;
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                if (i == 0 && j == 0)
                {
                    break; // comparing the pixel to itself
                }
                else if (deviceInput[(y + i) * width + (x + j)] == STRONG_EDGE)
                {
                    strongNeighbor = true;
                }
            }
        }
        if (strongNeighbor)
        {
            deviceOutput[y * width + x] = STRONG_EDGE;
        }
        else
        {
            deviceOutput[y * width + x] = NON_EDGE;
        }
    }
}



// This method is responsible for handling memory and calling the kernel for
// the final step of hysteresis thresholding
void hysteresisCuda(const cv::Mat& hostInput, cv::Mat& hostOutput)
{
    // Allocate memory on device for input and output
    unsigned char* deviceInput;
    unsigned char* deviceOutput;
    int bytes = hostInput.rows * hostInput.cols * sizeof(unsigned char);
    cudaMalloc((void**)&deviceInput, bytes);
    cudaMalloc((void**)&deviceOutput, bytes);

    // Copy memory from host to device 
    cudaMemcpy(deviceInput, hostInput.ptr(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutput, hostInput.ptr(), bytes, cudaMemcpyHostToDevice);

    // Call the kernel
    const dim3 numBlocks(ceil(hostInput.cols / BLOCK_SIZE),
        ceil(hostInput.rows / BLOCK_SIZE), 1);
    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    hysteresisKernel << <numBlocks, threadsPerBlock >> > (deviceInput, 
        deviceOutput, hostInput.cols, hostInput.rows);

    // Copy memory back to host after kernel is complete
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput.ptr(), deviceOutput, bytes, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}



// This method is our CPU powered BFS implementation of the second portion of
// hysteresis thresholding
void hysteresisCPU(cv::Mat& hostInput, cv::Mat& hostOutput) {

    std::queue<std::pair<int, int>> strongEdges;

    // populate queue with original strong edges
    for (int col = 0; col < hostInput.cols; col++) {
        for (int row = 0; row < hostInput.rows; row++) {
            if (hostInput.at<uchar>(row, col) == 255) {
                strongEdges.push(std::pair<int, int>(row, col));
            }
            else {
                hostOutput.at<uchar>(row, col) = 0;
            }
        }
    }

    int col;
    int row;
    // bfs from each strong edge
    while (!strongEdges.empty()) {
        row = strongEdges.front().first;
        col = strongEdges.front().second;
        strongEdges.pop();
        hostOutput.at<uchar>(row, col) = 255;

        // examine all neighbors, after checking if they are contained within
        // the image boundaries
        std::vector<std::pair<int, int>> neighbors;
        if (col - 1 > 0 && row - 1 > 0) { neighbors.push_back
            (std::pair<int, int>(row - 1, col - 1)); }
        if (col - 1 > 0) { neighbors.push_back
            (std::pair<int, int>(row, col - 1)); }
        if (col - 1 > 0 && row + 1 < hostInput.rows) { neighbors.push_back
            (std::pair<int, int>(row + 1, col - 1)); }
        if (row - 1 > 0) { neighbors.push_back(std::pair<int, int>
            (row - 1, col)); }
        if (row + 1 < hostInput.rows) { neighbors.push_back
            (std::pair<int, int>(row + 1, col)); }
        if (col + 1 < hostInput.cols && row - 1 > 0) { neighbors.push_back
            (std::pair<int, int>(row - 1, col + 1)); }
        if (col + 1 < hostInput.cols) { neighbors.push_back(std::pair<int, int>
            (row, col + 1)); }
        if (col + 1 < hostInput.cols && row + 1 < hostInput.rows) { 
            neighbors.push_back(std::pair<int, int>(row + 1, col + 1)); }

        // For each neighbor, if it is a weak edge, make it a strong edge and 
        // queue it 
        for (int i = 0; i < neighbors.size(); i++) {
            if (hostInput.at<uchar>(neighbors[i].first, neighbors[i].second) == 
                128) {
                strongEdges.push(std::pair<int, int>(neighbors[i].first, 
                    neighbors[i].second));
                hostInput.at<uchar>(neighbors[i].first, neighbors[i].second) = 
                    255;
            }
        }

    }
}




// This method is the kernel for the first step of hysteresis thresholding
// populating the output image with only non, weak, and strong edge indicators
__global__ void thresholdingKernel(unsigned char* deviceInput, 
    unsigned char* deviceOutput, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char inputValue = deviceInput[y * width + x];


    // if the value is below hystLow, color pixel as black
    if (inputValue < HYST_LOW) {
        deviceOutput[y * width + x] = NON_EDGE;
    }


    // if the value is at or above hystHigh, color pixel white
    if (inputValue >= HYST_HIGH) {
        deviceOutput[y * width + x] = STRONG_EDGE;
    }

    // if the value is at or above hystLow, but below hystHigh, color it white
    // only if it is connected to another edge pixel.
    if (inputValue < HYST_HIGH && inputValue >= HYST_LOW) {
        // this is a maybe pixel
        deviceOutput[y * width + x] = WEAK_EDGE;
    }

}



// This method is responsible for handling memory and calling the kernel that 
// does the first portion of hysteresis thresholding
void thresholdingCuda(const cv::Mat& hostInput, cv::Mat& hostOutput) {
    // Allocate memory on device for input and output
    unsigned char* deviceInput;
    unsigned char* deviceOutput;
    int bytes = hostInput.rows * hostInput.cols * sizeof(unsigned char);
    cudaMalloc((void**)&deviceInput, bytes);
    cudaMalloc((void**)&deviceOutput, bytes);

    // Copy memory from host to device 
    cudaMemcpy(deviceInput, hostInput.ptr(), bytes, cudaMemcpyHostToDevice);

    // Call the kernel
    const dim3 numBlocks(ceil(hostInput.cols / BLOCK_SIZE),
        ceil(hostInput.rows / BLOCK_SIZE), 1);
    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    thresholdingKernel << <numBlocks, threadsPerBlock >> > (deviceInput, 
        deviceOutput, hostInput.cols, hostInput.rows);

    // Copy memory back to host after kernel is complete
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput.ptr(), deviceOutput, bytes, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}



// This method is the kernel that handles non-maxima suppression
__global__ void nonMaximaSuppressionKernel(unsigned char* deviceInput, 
    unsigned char* deviceOutput, float* angles, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    bool isMaximum = false;    
    unsigned char intensity = deviceInput[y * width + x];
    float radians = angles[y * width + x];
    float degrees = radians * 180.0 / PI;

    if ((0 <= degrees && degrees < 22.5) || (157.5 <= degrees && degrees <= 180))
    {
        if (intensity > deviceInput[(y + 1) * width + x] &&
            intensity > deviceInput[(y - 1) * width + x])
        {
            isMaximum = true;
        }
    }
    else if (degrees <= 22.5 && degrees < 67.5)
    {
        if (intensity > deviceInput[(y - 1) * width + (x + 1)] &&
            intensity > deviceInput[(y + 1) * width + (x - 1)])
        {
            isMaximum = true;
        }
    }
    else if (degrees <= 67.5 && degrees < 112.5)
    {
        if (intensity > deviceInput[y * width + (x + 1)] &&
            intensity > deviceInput[y * width + (x - 1)])
        {
            isMaximum = true;
        }
    }
    else if (degrees <= 112.5 && degrees < 157.5)
    {
        if (intensity > deviceInput[(y - 1) * width + (x - 1)] &&
            intensity > deviceInput[(y + 1) * width + (x + 1)])
        {
            isMaximum = true;
        }
    }
    
    if (!isMaximum)
    {
        deviceOutput[y * width + x] = 0;
    }
}



// This method is responsible for memory managment and calling the kernel
// that performs the non-maxima suppression on a frame
void nonMaximaSuppressionCuda(const cv::Mat& hostInput, cv::Mat& hostOutput, 
    float* hostAngles)
{
    // Allocate memory on device for input and output
    unsigned char* deviceInput;
    unsigned char* deviceOutput;
    int bytes = hostInput.rows * hostInput.cols * sizeof(unsigned char);
    cudaMalloc((void**)&deviceInput, bytes);
    cudaMalloc((void**)&deviceOutput, bytes);

    // Allocate memory on device to store the angles of each edge
    float* deviceAngles;
    int anglesBytes = hostInput.rows * hostInput.cols * sizeof(float);
    cudaMalloc((void**)&deviceAngles, anglesBytes);

    // Copy memory from host to device 
    cudaMemcpy(deviceInput, hostInput.ptr(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutput, hostInput.ptr(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceAngles, hostAngles, anglesBytes, cudaMemcpyHostToDevice);

    // Call the kernel to apply non-maxima suppression
    const dim3 numBlocks(ceil(hostInput.cols / BLOCK_SIZE), 
        ceil(hostInput.rows / BLOCK_SIZE), 1);
    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    nonMaximaSuppressionKernel << <numBlocks, threadsPerBlock >> >(deviceInput, 
        deviceOutput, deviceAngles, hostInput.cols, hostInput.rows);

    // Copy memory back to host after kernel is complete
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput.ptr(), deviceOutput, bytes, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceAngles);
}



// This kernel creates an intensity gradient using the sobel operator
__global__ void sobelKernel(unsigned char* deviceInput, 
    unsigned char* deviceOutput, float* deviceAngles, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int gx = 0; 
    int gy = 0;

    // no need to calculate sobel[1, 4, 7] because all those values will be 0
    gx = (sobel_x[0] * deviceInput[(y - 1) * width + (x - 1)]) +
         (sobel_x[2] * deviceInput[(y - 1) * width + (x + 1)]) +
         (sobel_x[3] * deviceInput[(  y  ) * width + (x - 1)]) +
         (sobel_x[5] * deviceInput[(  y  ) * width + (x + 1)]) +
         (sobel_x[6] * deviceInput[(y + 1) * width + (x - 1)]) +
         (sobel_x[8] * deviceInput[(y + 1) * width + (x + 1)]);

    // no need to calculate sobel[3:5] because all those values will be 0
    gy = (sobel_y[0] * deviceInput[(y - 1) * width + (x - 1)]) +
         (sobel_y[1] * deviceInput[(y - 1) * width + (  x  )]) +
         (sobel_y[2] * deviceInput[(y - 1) * width + (x + 1)]) +
         (sobel_y[6] * deviceInput[(y + 1) * width + (x - 1)]) +
         (sobel_y[7] * deviceInput[(y + 1) * width + (  x  )]) +
         (sobel_y[8] * deviceInput[(y + 1) * width + (x + 1)]);

    deviceOutput[y * width + x] = 
        static_cast<unsigned char>(sqrt((float)(gx * gx) + (gy * gy)));
    deviceAngles[y * width + x] = atan((float) (gy / gx));
}



// This method is responsible for managing memory and calling the kernel
// that creates an intensity gradient using the sobel operator
void sobelCuda(const cv::Mat& hostInput, cv::Mat& hostOutput, float* hostAngles)
{
    // Allocate memory on device for input and output
    unsigned char* deviceInput;
    unsigned char* deviceOutput;
    int bytes = hostInput.rows * hostInput.cols * sizeof(unsigned char);
    cudaMalloc((void**)&deviceInput, bytes);
    cudaMalloc((void**)&deviceOutput, bytes);

    // Allocate memory on device to store the angles of each edge
    float* deviceAngles;
    int anglesBytes = hostInput.rows * hostInput.cols * sizeof(float);
    cudaMalloc((void**)&deviceAngles, anglesBytes);

    // Copy memory from host to device 
    cudaMemcpy(deviceInput, hostInput.ptr(), bytes, cudaMemcpyHostToDevice);

    // Populate the global memory symbols with Sobel kernel values
    int h_sobel_x[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int h_sobel_y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    cudaMemcpyToSymbol(sobel_x, h_sobel_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(sobel_y, h_sobel_y, 9 * sizeof(int));

    // Call the kernel to apply the Sobel filter
    const dim3 numBlocks(ceil(hostInput.cols / BLOCK_SIZE), ceil(hostInput.rows 
        / BLOCK_SIZE), 1);
    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    sobelKernel << <numBlocks, threadsPerBlock >> > (deviceInput, deviceOutput, 
        deviceAngles, hostInput.cols, hostInput.rows);

    // Copy memory back to host after kernel is complete
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput.ptr(), deviceOutput, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostAngles, deviceAngles, anglesBytes, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceAngles);
}



// This kernel performs a gaussian blur on the image deviceInput and outputs it
// to deviceOutput
__global__ void gaussianKernel(unsigned char* deviceInput, unsigned char* 
    deviceOutput, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sum = gaussian[0] * deviceInput[(y - 1) * width + (x - 1)] +
              gaussian[1] * deviceInput[(y - 1) * width + (  x  )] +
              gaussian[2] * deviceInput[(y - 1) * width + (x + 1)] +
              gaussian[3] * deviceInput[(  y  ) * width + (x - 1)] +
              gaussian[4] * deviceInput[(  y  ) * width + (  x  )] +
              gaussian[5] * deviceInput[(  y  ) * width + (x + 1)] +
              gaussian[6] * deviceInput[(y + 1) * width + (x - 1)] +
              gaussian[7] * deviceInput[(y + 1) * width + (  x  )] +
              gaussian[8] * deviceInput[(y + 1) * width + (x + 1)];

    int weights = gaussian[0] + gaussian[1] + gaussian[2] +
                  gaussian[3] + gaussian[4] + gaussian[5] +
                  gaussian[6] + gaussian[7] + gaussian[8];

    deviceOutput[y * width + x] = static_cast<unsigned char>(sum / weights);
}



// This method is responsible for managing memory and calling the kernel that
// performs gaussian blur
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
    const dim3 numBlocks(ceil(hostInput.cols / BLOCK_SIZE), ceil(hostInput.rows 
        / BLOCK_SIZE), 1);
    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    gaussianKernel << < numBlocks, threadsPerBlock >> > (deviceInput,
        deviceOutput, hostInput.cols, hostInput.rows); 

    // Copy memory back to host after kernel is complete
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput.ptr(), deviceOutput, bytes, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}


// This kernel is responsible for converting the color frame into a greyscale
// image, and outputing it to grayOutput
__global__ void grayscaleKernel(unsigned char* rgbInput, unsigned char* 
    grayOutput, int width, int height, int colorWidthStep, int grayWidthStep)
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



// This method is responsible for managing memory and calling the kernel that
// performs the coversion from color to a greyscale image
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
    const dim3 gridSize((hostInput.cols + blockSize.x - 1) / blockSize.x, 
        (hostInput.rows + blockSize.y - 1) / blockSize.y, 1); 
    grayscaleKernel << <gridSize, blockSize >> > (deviceInput, deviceOutput, 
        hostInput.cols, hostInput.rows, hostInput.step, hostOutput.step);

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
        }
        cap.release();
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
        std::cerr << "Not enough lines detected with hough transform" 
            << std::endl;
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
        // Anything over 1.5707f is a left lane candidate, but to remove errant 
        // vertical lines, you don't want to consider anything above 2.8416f
        if (theta >= 1.5707f && theta < 2.8416f) {
            difference = 3.14f - theta;
            // if true you have found a better left candidate
            if (difference < leftCandidateLeastDifference) {
                leftLaneCandidate = i;
                leftCandidateLeastDifference = difference;
            }
        }
        // looking at right lane candidate
        // Anything under 1.5707f is a right lane candidate, but to remove 
        // errant vertical lines, you don't want to consider anything below 0.3f
        if (theta < 1.5707f && theta > 0.3f) {
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
    for (size_t i = 0; i < lanes.size(); i++)
    {
        // elements of this polar coordinate line
        float rho = lanes[i][0], theta = lanes[i][1];
        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;

        // calculate bottom point to be drawn using x intercept with line
        // y = frame.rows - 1
        cv::Point pta;
        pta.x = cvRound(x0 - ((frame.rows - 1 - y0) / a) * b);
        pta.y = frame.rows - 1;

        // calculate bottom point to be drawn using x intercept with line
        // y = (4 * frame.rows) / 7
        cv::Point ptb;
        ptb.x = cvRound(x0 - ((((4 * frame.rows) / 7) - y0) / a) * b);
        ptb.y = (4 * frame.rows) / 7;

        // draw the line
        line(output, pta, ptb, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

        // testing line to show theta values of final lines
        //std::cerr << "Theta = " << theta << std::endl;
    }
    return output;
}



// This version of our GPU canny edge detetion path is an attempt at optimizing
// the process by keeping images in device memory and reducing copies from
// host to device and back.
cv::Mat gpuOptimized(const cv::Mat &frame, bool debug)
{
    int width = frame.cols;
    int height = frame.rows;
    int cols = frame.cols;
    int rows = frame.rows;
    cv::Mat output = cv::Mat(height, width, CV_8UC1);
    int rgb_bytes = frame.step * frame.rows;
    int bytes = rows * cols * sizeof(unsigned char);

    /*******************************************************************
    *    RGB TO GRAYSCALE CONVERSION
    *******************************************************************/
    unsigned char* grayscaleInput;
    unsigned char* grayscaleOutput;
    cudaMalloc(&grayscaleInput, rows * cols * sizeof(unsigned char) * CHANNELS);
    cudaMalloc(&grayscaleOutput, bytes);
    cudaMemcpy(grayscaleInput, frame.ptr(), rgb_bytes, cudaMemcpyHostToDevice);
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);
    grayscaleKernel << <gridSize, blockSize >> > (grayscaleInput, grayscaleOutput, width, height, frame.step, output.step);
    cudaDeviceSynchronize();
    if (debug)
    {
        cv::Mat grayscale = cv::Mat(height, width, CV_8UC1);
        cudaMemcpy(grayscale.ptr(), grayscaleOutput, bytes, cudaMemcpyDeviceToHost);
        cv::imshow("OPTIMIZED Grayscale", grayscale);
        cv::waitKey(0);
    }

    /*******************************************************************
    *    GAUSSIAN BLUR
    *******************************************************************/
    unsigned char* gaussianInput;
    cudaMalloc(&gaussianInput, bytes);
    cudaMemcpy(gaussianInput, grayscaleOutput, bytes, cudaMemcpyDeviceToDevice);
    unsigned char* gaussianOutput;
    cudaMalloc(&gaussianOutput, bytes);
    int hostGaussian[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    cudaMemcpyToSymbol(gaussian, hostGaussian, 9 * sizeof(int));
    const dim3 numBlocks(ceil(cols / BLOCK_SIZE), ceil(rows / BLOCK_SIZE), 1);
    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    gaussianKernel << < numBlocks, threadsPerBlock >> > (gaussianInput, gaussianOutput, cols, rows);
    cudaDeviceSynchronize();
    cudaFree(gaussianInput);
    if (debug)
    {
        cv::Mat gaussian = cv::Mat(height, width, CV_8UC1);
        //cudaDeviceSynchronize();
        cudaMemcpy(gaussian.ptr(), gaussianOutput, bytes, cudaMemcpyDeviceToHost);
        cv::imshow("OPTIMIZED Gaussian", gaussian);
        cv::waitKey(0);
    }

    
    /*******************************************************************
    *    SOBEL OPERATOR
    *******************************************************************/
    unsigned char* sobelInput;
    cudaMalloc(&sobelInput, bytes);
    cudaMemcpy(sobelInput, gaussianOutput, bytes, cudaMemcpyDeviceToDevice);
    unsigned char* sobelOutput;
    cudaMalloc(&sobelOutput, bytes);
    float* sobelAngles;
    cudaMalloc(&sobelAngles, rows * cols * sizeof(float));
    int h_sobel_x[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
    int h_sobel_y[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
    cudaMemcpyToSymbol(sobel_x, h_sobel_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(sobel_y, h_sobel_y, 9 * sizeof(int));   
    sobelKernel << <numBlocks, threadsPerBlock >> > (sobelInput, sobelOutput, sobelAngles, cols, rows);
    cudaDeviceSynchronize();
    cudaFree(gaussianOutput);
    cudaFree(sobelInput);
    if (debug)
    {
        cv::Mat sobel = cv::Mat(height, width, CV_8UC1);
        //cudaDeviceSynchronize();
        cudaMemcpy(sobel.ptr(), sobelOutput, bytes, cudaMemcpyDeviceToHost);
        cv::imshow("OPTIMIZED Sobel", sobel);
        cv::waitKey(0);
    }


    /*******************************************************************
    *    NON-MAXIMA SUPPRESSION
    *******************************************************************/
    unsigned char* nmsInput;
    cudaMalloc(&nmsInput, bytes);
    cudaMemcpy(nmsInput, sobelOutput, bytes, cudaMemcpyDeviceToDevice);
    unsigned char* nmsOutput;
    cudaMalloc(&nmsOutput, bytes);
    cudaMemcpy(nmsOutput, nmsInput, bytes, cudaMemcpyDeviceToDevice);
    float* nmsAngles;
    cudaMalloc(&nmsAngles, rows * cols * sizeof(float));
    cudaMemcpy(nmsAngles, sobelAngles, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    nonMaximaSuppressionKernel << <numBlocks, threadsPerBlock >> > (nmsInput, nmsOutput, nmsAngles, cols, rows);
    cudaDeviceSynchronize();
    cudaFree(sobelOutput);
    cudaFree(sobelAngles);
    cudaFree(nmsInput);
    if (debug)
    {
        cv::Mat nms = cv::Mat(height, width, CV_8UC1);
        cudaMemcpy(nms.ptr(), nmsOutput, bytes, cudaMemcpyDeviceToHost);
        cv::imshow("OPTIMIZED nms", nms);
        cv::waitKey(0);
    }


    /*******************************************************************
    *    HYSTERESIS THESHOLD - STAGE 1
    *******************************************************************/
    unsigned char* thresholdInput;
    cudaMalloc(&thresholdInput, bytes);
    cudaMemcpy(thresholdInput, nmsOutput, bytes, cudaMemcpyDeviceToDevice);
    unsigned char* thresholdOutput;
    cudaMalloc(&thresholdOutput, bytes);
    thresholdingKernel << <numBlocks, threadsPerBlock >> > (thresholdInput, thresholdOutput, cols, rows);
    cudaDeviceSynchronize();
    cudaFree(nmsOutput);
    cudaFree(nmsAngles);
    cudaFree(thresholdInput);
    if (debug)
    {
        cv::Mat threshold = cv::Mat(height, width, CV_8UC1);
        cudaMemcpy(threshold.ptr(), thresholdOutput, bytes, cudaMemcpyDeviceToHost);
        cv::imshow("OPTIMIZED threshold", threshold);
        cv::waitKey(0);
    }


    /*******************************************************************
    *    HYSTERESIS THESHOLD - STAGE 2
    *******************************************************************/
    unsigned char* hysteresisInput;
    cudaMalloc(&hysteresisInput, bytes);
    cudaMemcpy(hysteresisInput, thresholdOutput, bytes, cudaMemcpyDeviceToDevice);
    unsigned char* hysteresisOutput;
    cudaMalloc(&hysteresisOutput, bytes);
    cudaMemcpy(hysteresisOutput, hysteresisInput, bytes, cudaMemcpyDeviceToDevice);
    hysteresisKernel << <numBlocks, threadsPerBlock >> > (hysteresisInput, hysteresisOutput, cols, rows);
    cudaDeviceSynchronize();
    cudaFree(thresholdOutput);
    cudaFree(hysteresisInput);
    if (debug)
    {
        cv::Mat hysteresis = cv::Mat(height, width, CV_8UC1);
        cudaMemcpy(hysteresis.ptr(), hysteresisOutput, bytes, cudaMemcpyDeviceToHost);
        cv::imshow("OPTIMIZED hysteresis", hysteresis);
        cv::waitKey(0);
    }
    
    cudaMemcpy(output.ptr(), hysteresisOutput, bytes, cudaMemcpyDeviceToHost);
    cudaFree(hysteresisOutput);
    cudaThreadExit();
    return output;   
}



// This method represents our GPU accelerated canny edge detection
// implementation. It handles calling the different methods for the various
// steps that make up that process, and showing intermediate images if in demo
cv::Mat gpuCanny(const cv::Mat &frame, bool demo) {
    cv::Mat image = frame.clone();
    const int rows = image.rows;
    const int cols = image.cols;
    //imshow("Extracted Frame", image);
    //cv::waitKey(0);

    // convert the image to grayscale 
    cv::Mat grayscale = cv::Mat(rows, cols, CV_8UC1);
    grayscaleCuda(image, grayscale);
    if (demo)
    {
        imshow("Grayscale Image", grayscale);
        cv::waitKey(0);
    }

    // apply the Gaussian filter
    cv::Mat blurred = cv::Mat(rows, cols, CV_8UC1);
    gaussianCuda(grayscale, blurred);
    if (demo)
    {
        imshow("Blurred Image", blurred);
        cv::waitKey(0);
    }
  
    // apply the Sobel operator
    cv::Mat sobel = cv::Mat(rows, cols, CV_8UC1);
    float* angles = (float*)malloc(rows * cols * sizeof(float));
    sobelCuda(blurred, sobel, angles);
    if (demo)
    {
        imshow("Intensity Gradient Image", sobel);
        cv::waitKey(0);
    }

    // apply non-maxima suppression
    cv::Mat nms = cv::Mat(rows, cols, CV_8UC1);
    nonMaximaSuppressionCuda(sobel, nms, angles);
    if (demo)
    {
        imshow("Non-Maxima Suppression Image", nms);
        cv::waitKey(0);
    }
  
    // perform hysteresis thresholding - stage 1
    cv::Mat threshold = cv::Mat(rows, cols, CV_8UC1);
    thresholdingCuda(nms, threshold);
    if (demo)
    {
        imshow("Hysteresis Thresholded Image - Stage 1", threshold);
        cv::waitKey(0);
    }

    // perform hysteresis thresholding - stage 2
    cv::Mat hysteresis = cv::Mat(rows, cols, CV_8UC1);
    hysteresisCuda(threshold, hysteresis);
    if (demo)
    {
        imshow("Hysteresis Thresholded Image - Stage 2", hysteresis);
        cv::waitKey(0);
    }
    
    return hysteresis;
}



// COMMAND LINE ARGUMENTS
// argv[0] = program name
// argv[1] = file path to video file
// argv[2] = expected inputs "CPU" or "GPU", determines which implementations to run
// argv[3] = expected inputs "demo" will tell the program to show output
int main(int argc, char** argv)
{
    // set run configurations from command line arguments
    std::string videoFilePath = argv[1];
    bool gpuAccelerated;
    std::cout << argv[2] << std::endl;
    std::string config = argv[2];
    if (config == "GPU")
    {
        std::cout << "=== GPU IMPLEMENTATION === " << std::endl;
        gpuAccelerated = true;
    }
    else if (config == "CPU")
    {
        std::cout << "=== CPU IMPLEMENTATION === " << std::endl;
        gpuAccelerated = false;
    }
    else
    {
        std::cerr << "Invalid command line arguments!" << std::endl;
        return -1;
    }

    bool demo = false;
    bool debug = false;
    if (argc > 3)
    {
        std::string demo_arg = argv[3];
        if (demo_arg == "demo")
        {
            demo = true;
        }
        else if (demo_arg == "debug")
        {
            debug = true;
        }
        else
        {
            std::cerr << "Invalid command line arguments!" << std::endl;
            return -1;
        }
    }
    
    // extract the video frames into a vector
    std::vector<cv::Mat> framesOutput;
    extractFrames(videoFilePath, framesOutput);

    // start timing for the total run time including hough
    auto totalStart = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds opencvTime(0);
    std::chrono::milliseconds gpuTime(0);
    std::chrono::milliseconds houghTime(0);
    // loop through each fram
    for (int i = 0; i < framesOutput.size(); i++)
    {
        // create Mat to hold the edges from canny edge detection
        cv::Mat edges;
        int size = framesOutput.size();

        // This section is for when using the opencvCanny() implementation 
        // path (non-GPU accelarated)
        if (!gpuAccelerated) {
            // start timing for OpenCV canny edge detection
            auto opencvFrameStart = std::chrono::high_resolution_clock::now();
            edges = opencvCanny(framesOutput[i]);
            auto opencvFrameEnd = std::chrono::high_resolution_clock::now();
            auto opencvFrameMs = std::chrono::duration_cast<std::chrono::milliseconds>(opencvFrameEnd - opencvFrameStart);
            opencvTime += opencvFrameMs;
        }

        // This section is for when using our own GPU accelerated path
        else {
            // start timing for GPU edge detecton
            auto gpuFrameStart = std::chrono::high_resolution_clock::now();
            if (demo == true)
            {
                edges = gpuCanny(framesOutput[i], demo);
            }
            else
            {
                edges = gpuOptimized(framesOutput[i], debug);
            }
            auto gpuFrameEnd = std::chrono::high_resolution_clock::now();
            auto gpuFrameMs = std::chrono::duration_cast<std::chrono::milliseconds>(gpuFrameEnd - gpuFrameStart);
            gpuTime += gpuFrameMs;
        }

        // perform hough transform, storing lines detected in houghLines vector 
        std::vector<cv::Vec2f> houghLines;
        auto houghStart = std::chrono::high_resolution_clock::now();
        houghTransform(edges, houghLines);
        auto houghEnd = std::chrono::high_resolution_clock::now();
        houghTime += std::chrono::duration_cast<std::chrono::milliseconds>(houghEnd - houghStart);
        
        if (demo)
        {
            imshow("lanes", drawLines(framesOutput[i], houghLines));
            cv::waitKey(0);
        }
    }
    
    // end the timing for total run time including hough (for this frame)
    auto totalEnd = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<float> totalTime = totalEnd - totalStart;
    auto totalMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart); 
    std::cout << "Total execution time: " << totalMilliseconds.count() << " milliseconds" << std::endl;

    if (gpuAccelerated)
    {
        std::cout << "GPU Canny execution time (CUDA Kernels): " << gpuTime.count() << " milliseconds" << std::endl;
        std::cout << "CPU Hough transform execution time: " << houghTime.count() << "milliseconds" << std::endl;
    }
    if (!gpuAccelerated)
    {
        std::cout << "CPU openCV::Canny() execution time: " << opencvTime.count() << " milliseconds" << std::endl;
        std::cout << "CPU Hough transform execution time: " << houghTime.count() << "milliseconds" << std::endl;
    }
    cv::destroyAllWindows();
    return 0;
}
