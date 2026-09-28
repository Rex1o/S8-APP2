#pragma once
#include <Images/Image.hpp>
#include <cassert>
#include <opencv2/opencv.hpp>

struct PackedData
{
    uchar* data = nullptr;
    size_t size = 0;
    float bitsPerPixel;
};

class ImageGray : public Image 
{
public:
    ImageGray() = default;

    ImageGray(cv::Mat image);
    ImageGray(const ImageGray& image);
    
    inline uchar GetPixel(uint x, uint y) const
    { 
        assert(m_OpenCVImage_.type() == CV_8UC1);
        return m_OpenCVImage_.at<uchar>(y, x);
    }

    ImageGray RescaleNearestNeigbor(uint newHeight, uint newWidth);
    
    ImageGray RescaleBilinear(uint newHeight, uint newWidth);
    
    ImageGray RescaleBicubic(uint newHeight, uint newWidth);

    void GetMeanAndSTDDev(float& mean, float& variance) const;
    void GetMinAndMax(uchar& min, uchar& max) const;
    double CalculatePSNR(const ImageGray& other) const;

private:
    static double CardinalCubic(double p0, double p1, double p2, double p3, double t);
};