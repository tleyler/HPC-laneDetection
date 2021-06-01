#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;


Mat opencvCanny(const Mat& frame) {

    Mat edgeDetectedFrame;
    Canny(frame, edgeDetectedFrame, 1, 3, 3);

    return edgeDetectedFrame;
}

int main()
{
    std::cout << "Hello World!\n";

    Mat frame = imread("testingFrame.jpg");
    imshow("Original Frame", frame);
    imshow ("Edge Detected Frame", opencvCanny(frame));
    waitKey(0);

    std::cout << "Made it to the end" << std::endl;
}


