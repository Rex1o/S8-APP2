#pragma once
#include <Images/Image.hpp>
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

    void GetMeanAndVariance(float& mean, float& variance) const;
    void GetMinAndMax(uchar& min, uchar& max) const;

private:
    friend class AdaptiveQuantizer;
    friend class BTCQuantizer;
    friend class DCTQuantizer;
    friend class DPCMQuantizer;
    friend class VectorQuantizer;
    friend class UniformQuantizer;
    friend class FowardAdaptiveGaussQuantizer;
    friend class JayantQuantizer;

    static double CardinalCubic(double p0, double p1, double p2, double p3, double t);
};