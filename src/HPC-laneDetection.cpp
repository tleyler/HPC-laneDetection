#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "ed_pixel.h"
#include "kernel.cuh"
#include <iostream>

using namespace cv;

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
Mat opencvCanny(const Mat& frame) {
    // changing this number effects the amount of edges that it detects. The 
    // larger the number is, the less it will detect, only picking up larger 
    // edges. Through some quick experimentation I settled on 75, but this can
    // be adjusted later if we need more or less edges.
    double edgeThreshold = 75.0;

    // this mat will hold the edges image
    Mat edgeDetectedFrame;
    Canny(frame, edgeDetectedFrame, edgeThreshold, edgeThreshold * 3.0, 3);

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

    // Apply and output opencvCanny() to each extracted frame
    std::vector<std::vector<pixel_t>> pixelValues;
    for (int i = 0; i < framesOutput.size(); i++)
    {
        // call some function to convert each element of frame<Mat> to 2D vector of pixel_t
        imshow("Edge Detected Frame", opencvCanny(framesOutput[i]));
        waitKey(0);
    }

    return 0;
}
