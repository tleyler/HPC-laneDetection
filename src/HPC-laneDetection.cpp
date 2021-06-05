#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "ed_pixel.h"
#include <iostream>
#include <vector>

using namespace cv;

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

// This function accepts a single frame and performs a hough transform on it 
// Returns a vector of the lines that were detected
void houghTransform(const Mat& frame, std::vector<Vec2f> &houghLines) {
    // create the houghLines vector with HoughLines method
    HoughLines(frame, houghLines, 1, CV_PI / 180, 150, 0, 0);
    return;
}

void drawLines(const Mat& frame, std::vector<Vec2f>& houghLines) {
    
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

        // create Mat to hold the edges from canny edge detection
        Mat edges = opencvCanny(framesOutput[i]);
        imshow("Edge Detected Frame", edges);
        waitKey(0);

        // perform hough transform, storing lines detected in houghLines vector
        std::vector<Vec2f> houghLines;
        houghTransform(edges, houghLines);


    }

    return 0;
}
