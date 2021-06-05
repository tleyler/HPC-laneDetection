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
    double edgeThreshold = 200.0;

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

Mat drawLines(const Mat& frame, std::vector<Vec2f>& houghLines) {
    Mat output = frame.clone();
    // Draw the lines
    // Loop for drawing the lines on frame pulled from houghlines.cpp in opencv tutorials
    for (size_t i = 0; i < houghLines.size(); i++)
    {
        float rho = houghLines[i][0], theta = houghLines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(output, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    return output;
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
        imshow("lanes", drawLines(framesOutput[i], houghLines));
        waitKey(0);

    }

    return 0;
}
