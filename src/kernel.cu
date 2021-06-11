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

    deviceOutput[y * width + x] = static_cast<unsigned char>(sqrt((float)(gx * gx) + (gy * gy)));
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
    const dim3 numBlocks(ceil(hostInput.cols / BLOCK_SIZE), ceil(hostInput.rows / BLOCK_SIZE), 1);
    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    gaussianKernel << < numBlocks, threadsPerBlock >> > (deviceInput, deviceOutput, hostInput.cols, hostInput.rows); 

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
    // Code for drawing lines on an image pulled from houghlines.cpp in opencv 
    // tutorials and adapted for our purpose
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


cv::Mat gpuCanny(const cv::Mat &frame) {
    cv::Mat image = frame.clone();
    const int rows = image.rows;
    const int cols = image.cols;
    //imshow("Extracted Frame", image);
    //cv::waitKey(0);

    // convert the image to grayscale 
    cv::Mat grayscale = cv::Mat(rows, cols, CV_8UC1);
    grayscaleCuda(image, grayscale);
    // VISUAL DEBUG: compare our implementation with openCV implementation
    /*
    cv::Mat opencv_grayscale = cv::Mat(rows, cols, CV_8UC1);
    cvtColor(image, opencv_grayscale, cv::COLOR_RGB2GRAY);
    imshow("Grayscale Image", grayscale);
    cv::waitKey(0);
    imshow("openCV Grayscale Image", opencv_grayscale);
    cv::waitKey(0);
    */

    // apply the Gaussian filter
    cv::Mat blurred = cv::Mat(rows, cols, CV_8UC1);
    gaussianCuda(grayscale, blurred);
    // VISUAL DEBUG: compare our implementation with openCV implementation
    /*
    cv::Mat opencv_blurred = cv::Mat(rows, cols, CV_8UC1);
    cv::GaussianBlur(grayscale, opencv_blurred, cv::Size(3, 3), 0);
    imshow("Blurred Image", blurred);
    cv::waitKey(0);
    imshow("openCV Blurred Image", opencv_blurred);
    cv::waitKey(0);
    */
  
    // apply the Sobel operator
    cv::Mat sobel = cv::Mat(rows, cols, CV_8UC1);
    sobelCuda(blurred, sobel);
    // VISUAL DEBUG: compare our implementation with openCV implementation
    /*
    cv::Mat opencv_sobel = cv::Mat(rows, cols, CV_8UC1);
    cv::Sobel(blurred, opencv_sobel, CV_8UC1, 1, 1); 
    imshow("Intensity Gradient Image", sobel);
    cv::waitKey();
    imshow("openCV Intensity Gradient", opencv_sobel);
    cv::waitKey();
    */
    

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

    bool gpuAccelerated = false;

    for (int i = 0; i < framesOutput.size(); i++)
    {
        cv::Mat edges;

        // This section is for when using the opencvCanny() implementation 
        // path (non-GPU accelarated)
        if (!gpuAccelerated) {
            // create Mat to hold the edges from canny edge detection
            edges = opencvCanny(framesOutput[i]);
            //imshow("Edge Detected Frame", edges);
            //cv::waitKey(0);
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
