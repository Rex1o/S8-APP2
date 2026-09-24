#pragma once
#include <Image.hpp>
#include <cassert>
#include <opencv2/opencv.hpp>

class ImageGray : public Image 
{
public:
    ImageGray() = default;

    ImageGray(cv::Mat image);
    ImageGray(const ImageGray& image);
    
    inline uchar GetPixel(uint x, uint y)
    { 
        assert(m_OpenCVImage_.type() == CV_8UC1);
        return m_OpenCVImage_.at<uchar>(y, x);
    }

    ImageGray RescaleNearestNeigbor(uint newHeight, uint newWidth);
    
    ImageGray RescaleBilinear(uint newHeight, uint newWidth);
    
    ImageGray RescaleBicubic(uint newHeight, uint newWidth);
private:
    static double CardinalCubic(double p0, double p1, double p2, double p3, double t);
};