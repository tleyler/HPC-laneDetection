#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;

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

int main()
{
    // ------------------------------------------------------------------------
    // Code in this section is just to pull, display, and manipulate a test
    // image frame until we have the capability for working with video all done
    
    // pull the test image and display it
    Mat frame = imread("testingFrame.jpg");
    imshow("Original Frame", frame);
    // also call the edge detection function that utilizes opencv Canny() and
    // display the output of that also
    imshow ("Edge Detected Frame", opencvCanny(frame));
    // wait for input, to give the user a chance to view images
    waitKey(0);

    // End test image frame section
    // ------------------------------------------------------------------------
}


