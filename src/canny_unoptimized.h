/*
 * canny_unoptimized.cu
 * This is a parallel unoptimized version of Canny Edge Detection in CUDA
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "canny_unoptimized.h"
// For Pi
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

#define _USE_MATH_DEFINES
#define RGB2GRAY_CONST_ARR_SIZE 3
#define STRONG_EDGE 255          // White single channel pixel value
#define NON_EDGE 0               // Black single channel pixel value

 //*****************************************************************************************
 // CUDA Gaussian Filter Implementation
 //*****************************************************************************************

// Apply a Gaussuan filter to blurr the image
__global__ void cu_apply_gaussian_filter(pixel_t* in_pixels, pixel_t* out_pixels, int rows, int cols, double* in_kernel) {
    // copy kernel array from global memory to a shared array
    __shared__ double kernel[KERNEL_SIZE][KERNEL_SIZE];
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            kernel[i][j] = in_kernel[i * KERNEL_SIZE + j];
        }
    }

    __syncthreads();

    // Determine id of thread which corresponds to an individual pixel
    int pixNum = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixNum >= 0 && pixNum < rows * cols) {
        double kernelSum;
        double redPixelVal;
        double greenPixelVal;
        double bluePixelVal;

        // Apply kernel to each pixel of image
        for (int i = 0; i < KERNEL_SIZE; ++i) {
            for (int j = 0; j < KERNEL_SIZE; ++j) {

                // check edge cases, if within bounds, apply filter
                if (((pixNum + ((i - ((KERNEL_SIZE - 1) / 2)) * cols) + j - ((KERNEL_SIZE - 1) / 2)) >= 0)
                    && ((pixNum + ((i - ((KERNEL_SIZE - 1) / 2)) * cols) + j - ((KERNEL_SIZE - 1) / 2)) <= rows * cols - 1)
                    && (((pixNum % cols) + j - ((KERNEL_SIZE - 1) / 2)) >= 0)
                    && (((pixNum % cols) + j - ((KERNEL_SIZE - 1) / 2)) <= (cols - 1))) {

                    int index = pixNum + ((i - ((KERNEL_SIZE - 1) / 2)) * cols) + j - ((KERNEL_SIZE - 1) / 2);
                    redPixelVal += kernel[j][j] * in_pixels[index].red;
                    greenPixelVal += kernel[i][j] * in_pixels[index].green;
                    bluePixelVal += kernel[i][j] * in_pixels[index].blue;
                    kernelSum += kernel[i][j];
                }
            }
        }

        // update output image
        out_pixels[pixNum].red = redPixelVal / kernelSum;
        out_pixels[pixNum].green = greenPixelVal / kernelSum;
        out_pixels[pixNum].blue = bluePixelVal / kernelSum;
    }
}


//*****************************************************************************************
// CUDA Intensity Gradient Implementation
//*****************************************************************************************

// Compute gradient (first order derivative x and y). This is the CUDA kernel for taking the
//    derivative of color contrasts in adjacent images.
__global__ void cu_compute_intensity_gradient(pixel_t* in_pixels, pixel_channel_t_signed* deltaX_channel,
    pixel_channel_t_signed* deltaY_channel, unsigned parser_length, unsigned offset) {

    // compute delta X
    // deltaX = f(x+1) - f(x-1)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // condition here skips first and last row
    if ((idx > offset) && (idx < (parser_length * offset) - offset)) {
        int16_t deltaXred = 0;
        int16_t deltaYred = 0;
        int16_t deltaXblue = 0;
        int16_t deltaYblue = 0;
        int16_t deltaXgreen = 0;
        int16_t deltaYgreen = 0;

        // first column
        if ((idx % offset) == 0) {
            // gradient at the first pixel of each line
            // note: at the edge pix[idx-1] does NOT exist
            deltaXred = (int16_t)(in_pixels[idx + 1].red - in_pixels[idx].red);
            deltaXgreen = (int16_t)(in_pixels[idx + 1].green - in_pixels[idx].green);
            deltaXblue = (int16_t)(in_pixels[idx + 1].blue - in_pixels[idx].blue);
            // gradient at the first pixel of each line
            // note: at the edge pix[idx-1] does NOT exist
            deltaYred = (int16_t)(in_pixels[idx + offset].red - in_pixels[idx].red);
            deltaYgreen = (int16_t)(in_pixels[idx + offset].green - in_pixels[idx].green);
            deltaYblue = (int16_t)(in_pixels[idx + offset].blue - in_pixels[idx].blue);
        }
        // last column
        else if ((idx % offset) == (offset - 1))
        {
            deltaXred = (int16_t)(in_pixels[idx].red - in_pixels[idx - 1].red);
            deltaXgreen = (int16_t)(in_pixels[idx].green - in_pixels[idx - 1].green);
            deltaXblue = (int16_t)(in_pixels[idx].blue - in_pixels[idx - 1].blue);
            deltaYred = (int16_t)(in_pixels[idx].red - in_pixels[idx - offset].red);
            deltaYgreen = (int16_t)(in_pixels[idx].green - in_pixels[idx - offset].green);
            deltaYblue = (int16_t)(in_pixels[idx].blue - in_pixels[idx - offset].blue);
        }
        // gradients where NOT edge
        else
        {
            deltaXred = (int16_t)(in_pixels[idx + 1].red - in_pixels[idx - 1].red);
            deltaXgreen = (int16_t)(in_pixels[idx + 1].green - in_pixels[idx - 1].green);
            deltaXblue = (int16_t)(in_pixels[idx + 1].blue - in_pixels[idx - 1].blue);
            deltaYred = (int16_t)(in_pixels[idx + offset].red - in_pixels[idx - offset].red);
            deltaYgreen = (int16_t)(in_pixels[idx + offset].green - in_pixels[idx - offset].green);
            deltaYblue = (int16_t)(in_pixels[idx + offset].blue - in_pixels[idx - offset].blue);
        }
        deltaX_channel[idx] = (int16_t)(0.2989 * deltaXred + 0.5870 * deltaXgreen + 0.1140 * deltaXblue);
        deltaY_channel[idx] = (int16_t)(0.2989 * deltaYred + 0.5870 * deltaYgreen + 0.1140 * deltaYblue);
    }
}


//*****************************************************************************************
// CUDA Gradient Magnitude Implementation
//*****************************************************************************************

// Compute magnitude of gradient (deltaX and deltaY) per pixel
__global__ void cu_magnitude(pixel_channel_t_signed* deltaX, pixel_channel_t_signed* deltaY,
    pixel_channel_t* out_pixel, unsigned parser_length, unsigned offset) {

    // computation 
    // assign a thread to each pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < parser_length * offset) {
        out_pixel[idx] = (pixel_channel_t)(sqrt((double)deltaX[idx] * deltaX[idx] + (double)deltaY[idx] * deltaY[idx]) + 0.5);
    }
}


//*****************************************************************************************
// CUDA Non Maximal Suppression Implementation
//*****************************************************************************************

// Non maximal suppression
// If the center pixel is not greater than neighbor pixels in the direction,
// then the center pixel is set to zero
// This process results in one pixel wide ridge
__global__ void cu_suppress_non_max(pixel_channel_t* mag, pixel_channel_t_signed* deltaX,
    pixel_channel_t_signed* deltaY, pixel_channel_t* nms, unsigned parser_length, unsigned offset) {

    // Suppressed value is 0
    const pixel_channel_t SUPPRESSED = 0;

    // Find the index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < parser_length * offset) {
        float alpha;
        float mag1, mag2;
        
        // put zero on all boundary of the image
        // Top edge line of the image
        if ((idx >= 0) && (idx < offset))
            nms[idx] = SUPPRESSED;
        
        // Bottom edge
        else if ((idx >= (parser_length - 1) * offset) && (idx < (offset * parser_length)))
            nms[idx] = SUPPRESSED;
        
        // Left and right edge line
        else if (((idx % offset) == 0 || ((idx % offset) == (offset - 1))))
            nms[idx] = SUPPRESSED;

        // Not the boundary
        else
        {
            // if magnitude = 0, no edge
            if (mag[idx] == 0)
                nms[idx] = SUPPRESSED;
            else {
                if (deltaX[idx] >= 0)
                {
                    if (deltaY[idx] >= 0) // dx >= 0, dy >= 0
                    {
                        if ((deltaX[idx] - deltaY[idx]) >= 0)      // direction 1 (South-East-East)
                        {
                            alpha = (float)deltaY[idx] / deltaX[idx];
                            mag1 = (1 - alpha) * mag[idx + 1] + alpha * mag[idx + offset + 1];
                            mag2 = (1 - alpha) * mag[idx - 1] + alpha * mag[idx - offset + 1];
                        }
                        else                                       // direction 2 (SSE)
                        {
                            alpha = (float)deltaX[idx] / deltaY[idx];
                            mag1 = (1 - alpha) * mag[idx + offset] + alpha * mag[idx + offset + 1];
                            mag2 = (1 - alpha) * mag[idx - offset] + alpha * mag[idx - offset - 1];
                        }
                    }
                    else  // dx >= 0, dy < 0
                    {
                        if ((deltaX[idx] + deltaY[idx]) >= 0)    // direction 8 (NEE)
                        {
                            alpha = (float)-deltaY[idx] / deltaX[idx];
                            mag1 = (1 - alpha) * mag[idx + 1] + alpha * mag[idx - offset + 1];
                            mag2 = (1 - alpha) * mag[idx - 1] + alpha * mag[idx + offset - 1];
                        }
                        else                                // direction 7 (NNE)
                        {
                            alpha = (float)deltaX[idx] / -deltaY[idx];
                            mag1 = (1 - alpha) * mag[idx + offset] + alpha * mag[idx + offset - 1];
                            mag2 = (1 - alpha) * mag[idx - offset] + alpha * mag[idx - offset + 1];
                        }
                    }
                }
                else
                {
                    if (deltaY[idx] >= 0) // dx < 0, dy >= 0
                    {
                        if ((deltaX[idx] + deltaY[idx]) >= 0)    // direction 3 (SSW)
                        {
                            alpha = (float)-deltaX[idx] / deltaY[idx];
                            mag1 = (1 - alpha) * mag[idx + offset] + alpha * mag[idx + offset - 1];
                            mag2 = (1 - alpha) * mag[idx - offset] + alpha * mag[idx - offset + 1];
                        }
                        else                                // direction 4 (SWW)
                        {
                            alpha = (float)deltaY[idx] / -deltaX[idx];
                            mag1 = (1 - alpha) * mag[idx - 1] + alpha * mag[idx + offset - 1];
                            mag2 = (1 - alpha) * mag[idx + 1] + alpha * mag[idx - offset + 1];
                        }
                    }

                    else // dx < 0, dy < 0
                    {
                        if ((-deltaX[idx] + deltaY[idx]) >= 0)   // direction 5 (NWW)
                        {
                            alpha = (float)deltaY[idx] / deltaX[idx];
                            mag1 = (1 - alpha) * mag[idx - 1] + alpha * mag[idx - offset - 1];
                            mag2 = (1 - alpha) * mag[idx + 1] + alpha * mag[idx + offset + 1];
                        }
                        else                                // direction 6 (NNW)
                        {
                            alpha = (float)deltaX[idx] / deltaY[idx];
                            mag1 = (1 - alpha) * mag[idx - offset] + alpha * mag[idx - offset - 1];
                            mag2 = (1 - alpha) * mag[idx + offset] + alpha * mag[idx + offset + 1];
                        }
                    }
                }

                // non-maximal suppression
                // compare mag1, mag2 and mag[t]
                // if mag[t] is smaller than one of the neighbours then suppress it
                if ((mag[idx] < mag1) || (mag[idx] < mag2))
                    nms[idx] = SUPPRESSED;
                else
                {
                    nms[idx] = mag[idx];
                }

            } // END OF ELSE (mag != 0)
        } // END OF FOR(j)
    } // END OF FOR(i)
}


//*****************************************************************************************
// CUDA Hysteresis Implementation
//*****************************************************************************************

// This is helper function that runs on the GPU
// It checks if the eight immediate neighbors of a pixel at a given index are above
// a low threshold, and if they are, sets them to strong edges. This effectively connects the edges.
__device__ void trace_immed_neighbors(pixel_channel_t* out_pixels, pixel_channel_t* in_pixels,
    unsigned idx, pixel_channel_t t_low, unsigned img_width) {

    // directions representing indices of neighbors
    unsigned n, s, e, w;
    unsigned nw, ne, sw, se;

    // get indices
    n = idx - img_width;
    nw = n - 1;
    ne = n + 1;
    s = idx + img_width;
    sw = s - 1;
    se = s + 1;
    w = idx - 1;
    e = idx + 1;

    if (in_pixels[nw] >= t_low)
        out_pixels[nw] = STRONG_EDGE;

    if (in_pixels[n] >= t_low)
        out_pixels[n] = STRONG_EDGE;

    if (in_pixels[ne] >= t_low)
        out_pixels[ne] = STRONG_EDGE;

    if (in_pixels[w] >= t_low)
        out_pixels[w] = STRONG_EDGE;

    if (in_pixels[e] >= t_low)
        out_pixels[e] = STRONG_EDGE;

    if (in_pixels[sw] >= t_low)
        out_pixels[sw] = STRONG_EDGE;

    if (in_pixels[s] >= t_low)
        out_pixels[s] = STRONG_EDGE;

    if (in_pixels[se] >= t_low)
        out_pixels[se] = STRONG_EDGE;
}


// CUDA implementation of Canny hysteresis high thresholding.
// This kernel is the first pass in the parallel hysteresis step.
// It launches a thread for every pixel and checks if the value of that pixel
// is above a high threshold. If it is, the thread marks it as a strong edge (set to 1)
// in a pixel map and sets the value to the channel max. If it is not, the thread sets
// the pixel map at the index to 0 and zeros the output buffer space at that index.
//
// The output of this step is a mask of strong edges and an output buffer with white values
// at the mask indices which are set.
__global__ void cu_hysteresis_high(pixel_channel_t* out_pixels, pixel_channel_t* in_pixels, 
    unsigned* strong_edge_mask, pixel_channel_t t_high, unsigned img_height, unsigned img_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (img_height * img_width)) {
        // apply high threshold
        if (in_pixels[idx] > t_high) {
            strong_edge_mask[idx] = 1;
            out_pixels[idx] = STRONG_EDGE;
        }
        else {
            strong_edge_mask[idx] = 0;
            out_pixels[idx] = NON_EDGE;
        }
    }
}


// CUDA implementation of Canny hysteresis low thresholding.
// This kernel is the second pass in the parallel hysteresis step. 
// It launches a thread for every pixel, but skips the first and last rows and columns.
// For surviving threads, the pixel at the thread ID index is checked to see if it was 
// previously marked as a strong edge in the first pass. If it was, the thread checks 
// their eight immediate neighbors and connects them (marks them as strong edges)
// if the neighbor is above the low threshold.
// The output of this step is an output buffer with both "strong" and "connected" edges
// set to white values. This is the final edge detected image.
__global__ void cu_hysteresis_low(pixel_channel_t* out_pixels, pixel_channel_t* in_pixels, 
    unsigned* strong_edge_mask, unsigned t_low, unsigned img_height, unsigned img_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx > img_width)                               /* skip first row */
        && (idx < (img_height * img_width) - img_width) /* skip last row */
        && ((idx % img_width) < (img_width - 1))        /* skip last column */
        && ((idx % img_width) > (0)))                  /* skip first column */
    {
        if (strong_edge_mask[idx] == 1) { /* if this pixel was previously found to be a strong edge */
            trace_immed_neighbors(out_pixels, in_pixels, idx, t_low, img_width);
        }
    }
}


//*****************************************************************************************
// Create a blur kernel
//*****************************************************************************************
void populate_blur_kernel(double out_kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    double scaleVal = 1;
    double stDev = (double)KERNEL_SIZE / 3;

    for (int i = 0; i < KERNEL_SIZE; ++i) {
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            double xComp = pow((i - KERNEL_SIZE / 2), 2);
            double yComp = pow((j - KERNEL_SIZE / 2), 2);

            double stDevSq = pow(stDev, 2);
            double pi = M_PI;

            //calculate the value at each index of the Kernel
            double kernelVal = exp(-(((xComp)+(yComp)) / (2 * stDevSq)));
            kernelVal = (1 / (sqrt(2 * pi) * stDev)) * kernelVal;

            //populate Kernel
            out_kernel[i][j] = kernelVal;

            if (i == 0 && j == 0)
            {
                scaleVal = out_kernel[0][0];
            }

            //normalize Kernel
            out_kernel[i][j] = out_kernel[i][j] / scaleVal;
        }
    }
}


//*****************************************************************************************
// Entry point for serial program calling CUDA implementation
//*****************************************************************************************

void cu_detect_edges(pixel_channel_t* final_pixels, pixel_t* orig_pixels, int rows, int cols)
{
    float totalTime = 0;
    float gaussianTime = 0;
    float gradientTime = 0;
    float magnitudeTime = 0;
    float nonMaxTime = 0;
    float hysteresisHighTime = 0;
    float hysteresisLowTime = 0;

    // kernel execution configuration parameters
    int num_blks = (rows * cols) / 1024;
    int thd_per_blk = 1024;
    int grid = 0;
    pixel_channel_t t_high = 30;           // High threshold for hysteresis
    pixel_channel_t t_low = t_high / 3;     // Low threshold ratio 3:1

    // Create a blur kernel
    double kernel[KERNEL_SIZE][KERNEL_SIZE];
    populate_blur_kernel(kernel);

    // device buffers
    pixel_t* in, * out;
    pixel_channel_t* single_channel_buf0;
    pixel_channel_t* single_channel_buf1;
    pixel_channel_t_signed* deltaX;
    pixel_channel_t_signed* deltaY;
    double* d_blur_kernel;
    unsigned* idx_map;

    // allocate device memory
    cudaMalloc((void**)&in, sizeof(pixel_t) * rows * cols);
    cudaMalloc((void**)&out, sizeof(pixel_t) * rows * cols);
    cudaMalloc((void**)&single_channel_buf0, sizeof(pixel_channel_t) * rows * cols);
    cudaMalloc((void**)&single_channel_buf1, sizeof(pixel_channel_t) * rows * cols);
    cudaMalloc((void**)&deltaX, sizeof(pixel_channel_t_signed) * rows * cols);
    cudaMalloc((void**)&deltaY, sizeof(pixel_channel_t_signed) * rows * cols);
    cudaMalloc((void**)&idx_map, sizeof(idx_map[0]) * rows * cols);
    cudaMalloc((void**)&d_blur_kernel, sizeof(d_blur_kernel[0]) * KERNEL_SIZE * KERNEL_SIZE);

    // data transfer image pixels to device
    cudaMemcpy(in, orig_pixels, rows * cols * sizeof(pixel_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blur_kernel, kernel, sizeof(d_blur_kernel[0]) * KERNEL_SIZE * KERNEL_SIZE, cudaMemcpyHostToDevice);

    //create cuda event variables for timing
    cudaEvent_t startAllKernels, endAllKernels, stGaussian, edGaussian, stGradient, edGradient, stMagnitude, edMagnitude, stNonMax, edNonMax, stHystHigh, edHystHigh, stHystLow, edHystLow;

    //init the cuda event objects
    cudaEventCreate(&startAllKernels);
    cudaEventCreate(&endAllKernels);
    cudaEventCreate(&stGaussian);
    cudaEventCreate(&edGaussian);
    cudaEventCreate(&stGradient);
    cudaEventCreate(&edGradient);
    cudaEventCreate(&stMagnitude);
    cudaEventCreate(&edMagnitude);
    cudaEventCreate(&stNonMax);
    cudaEventCreate(&edNonMax);
    cudaEventCreate(&stHystHigh);
    cudaEventCreate(&edHystHigh);
    cudaEventCreate(&stHystLow);
    cudaEventCreate(&edHystLow);

    cudaEventRecord(startAllKernels, 0);


    // run canny edge detection core - CUDA kernels
    // use streams to ensure the kernels are in the same task
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEventRecord(stGaussian, 0);
    cu_apply_gaussian_filter << <num_blks, thd_per_blk, grid, stream >> > (in, out, rows, cols, d_blur_kernel);
    cudaEventRecord(edGaussian, 0);
    cudaEventSynchronize(edGaussian);

    cudaEventRecord(stGradient, 0);
    cu_compute_intensity_gradient << <num_blks, thd_per_blk, grid, stream >> > (out, deltaX, deltaY, rows, cols);
    cudaEventRecord(edGradient, 0);
    cudaEventSynchronize(edGradient);

    cudaEventRecord(stMagnitude, 0);
    cu_magnitude << <num_blks, thd_per_blk, grid, stream >> > (deltaX, deltaY, single_channel_buf0, rows, cols);
    cudaEventRecord(edMagnitude, 0);
    cudaEventSynchronize(edMagnitude);

    cudaEventRecord(stNonMax, 0);
    cu_suppress_non_max << <num_blks, thd_per_blk, grid, stream >> > (single_channel_buf0, deltaX, deltaY, single_channel_buf1, rows, cols);
    cudaEventRecord(edNonMax, 0);
    cudaEventSynchronize(edNonMax);

    cudaEventRecord(stHystHigh, 0);
    cu_hysteresis_high << <num_blks, thd_per_blk, grid, stream >> > (single_channel_buf0, single_channel_buf1, idx_map, t_high, rows, cols);
    cudaEventRecord(edHystHigh, 0);
    cudaEventSynchronize(edHystHigh);

    cudaEventRecord(stHystLow, 0);
    cu_hysteresis_low << <num_blks, thd_per_blk, grid, stream >> > (single_channel_buf0, single_channel_buf1, idx_map, t_low, rows, cols);
    cudaEventRecord(edHystLow, 0);
    cudaEventSynchronize(edHystLow);

    cudaEventRecord(endAllKernels, 0);
    cudaEventSynchronize(endAllKernels);

    // wait for everything to finish
    cudaDeviceSynchronize();

    // copy result back to the host
    cudaMemcpy(final_pixels, single_channel_buf0, rows * cols * sizeof(pixel_channel_t), cudaMemcpyDeviceToHost);

    //calculate elapsed time in milliseconds
    cudaEventElapsedTime(&totalTime, startAllKernels, endAllKernels);
    cudaEventElapsedTime(&gaussianTime, stGaussian, edGaussian);
    cudaEventElapsedTime(&gradientTime, stGradient, edGradient);
    cudaEventElapsedTime(&magnitudeTime, stMagnitude, edMagnitude);
    cudaEventElapsedTime(&nonMaxTime, stNonMax, edNonMax);
    cudaEventElapsedTime(&hysteresisHighTime, stHystHigh, stHystLow);
    cudaEventElapsedTime(&hysteresisLowTime, stHystLow, edHystLow);


    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Total Canny Time: " << totalTime << " ms" << std::endl;
    /*std::cout << "Gaussian Time: " << gaussianTime << " ms" << std::endl;
    std::cout << "Gradient Time: " << gradientTime << " ms" << std::endl;
    std::cout << "Magnitude Time: " << magnitudeTime << " ms" << std::endl;
    std::cout << "Non-Max Suppression Time: " << nonMaxTime << " ms" << std::endl;
    std::cout << "Hysteresis High-Threshold Time: " << hysteresisHighTime << " ms" << std::endl;
    std::cout << "Hysteresis Low-Threshold Time: " << hysteresisLowTime << " ms" << std::endl;*/

    std::cout << std::endl;
    std::cout << std::endl;

    float numOpsGaussian = (rows * cols) * (KERNEL_SIZE * KERNEL_SIZE);
    float numOps = rows * cols;
    float gFlopsGaussian = 1.0e-9 * numOpsGaussian / gaussianTime;
    float gFlopsGradient = 1.0e-9 * numOps / gradientTime;
    float gFlopsMagnitude = 1.0e-9 * numOps / magnitudeTime;
    float gFlopsNonMaxSupression = 1.0e-9 * numOps / nonMaxTime;
    float gFlopshysteresisHigh = 1.0e-9 * numOps / hysteresisHighTime;
    float gFlopshysteresisLow = 1.0e-9 * numOps / hysteresisLowTime;
    float gFlopsTotalCanny = gFlopsGaussian + gFlopsGradient + gFlopsMagnitude + gFlopsNonMaxSupression + gFlopshysteresisHigh + gFlopshysteresisLow;
    float gFlopsTotalCannyAlt = 1.0e-9 * (numOpsGaussian + (numOps * 5)) / totalTime;

    /* std::cout << "gflops gaussian: " << gFlopsGaussian << std::endl;
     std::cout << "gflops gradient: " << gFlopsGradient << std::endl;
     std::cout << "gflops magnitude: " << gFlopsMagnitude << std::endl;
     std::cout << "gflops nonMaxSuppression: " << gFlopsNonMaxSupression << std::endl;
     std::cout << "gflops hysteresis high: " << gFlopshysteresisHigh << std::endl;
     std::cout << "gflops hysteresis low: " << gFlopshysteresisLow << std::endl;*/
    std::cout << "gflops canny: " << gFlopsTotalCanny << std::endl;
    //std::cout << "gflops canny alt: " << gFlopsTotalCannyAlt << std::endl;



    //free cuda event objects
    cudaEventDestroy(startAllKernels);
    cudaEventDestroy(endAllKernels);
    cudaEventDestroy(stGaussian);
    cudaEventDestroy(edGaussian);
    cudaEventDestroy(stGradient);
    cudaEventDestroy(edGradient);
    cudaEventDestroy(stMagnitude);
    cudaEventDestroy(edMagnitude);
    cudaEventDestroy(stNonMax);
    cudaEventDestroy(edNonMax);
    cudaEventDestroy(stHystHigh);
    cudaEventDestroy(edHystHigh);
    cudaEventDestroy(stHystLow);
    cudaEventDestroy(edHystLow);

    // cleanup
    cudaFree(in);
    cudaFree(out);
    cudaFree(single_channel_buf0);
    cudaFree(single_channel_buf1);
    cudaFree(deltaX);
    cudaFree(deltaY);
    cudaFree(idx_map);
    cudaFree(d_blur_kernel);
}

