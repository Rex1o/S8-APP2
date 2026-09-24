#include <ImageGray.hpp>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <algorithm>

ImageGray::ImageGray(cv::Mat image)
{
    assert(m_OpenCVImage_.type() == CV_8UC1);
    m_OpenCVImage_ = image;
}

ImageGray::ImageGray(const ImageGray &image)
{
    m_OpenCVImage_ = image.m_OpenCVImage_.clone();
}

ImageGray ImageGray::RescaleNearestNeigbor(uint newHeight, uint newWidth)
{
    assert(newHeight != 0 && newWidth != 0);

    if (GetHeight() == newHeight && GetWidth() == newWidth)
    {
        return ImageGray(*this);
    }

    cv::Mat newImage(newHeight, newWidth, CV_8UC1);
    
    double xScaleFactor =  static_cast<double>(GetWidth()) / static_cast<double>(newWidth);
    double yScaleFactor =  static_cast<double>(GetHeight()) / static_cast<double>(newHeight); 
    
    double maxWidthIndexDouble = static_cast<double>(GetWidth() - 1);
    double maxHeightIndexDouble = static_cast<double>(GetHeight() - 1);
    // Pixels in opencv are contiguous in the rows so start with y as the for loop
    // this increases the cache locality
    for (size_t y = 0; y < newHeight; y++)
    {
        double ySource = (static_cast<double>(y) + 0.5) * yScaleFactor;
        uint yIndex = static_cast<uint>(std::round(ySource));
        
        for (size_t x = 0; x < newWidth; x++)
        {
            double xSource = (static_cast<double>(x) + 0.5) * xScaleFactor - 0.5;
            uint xIndex = static_cast<uint>(std::round(xSource)); // We round to ensure no values a clippped so 2.7 would instead give 3
            // Apply pixel value
            newImage.at<uchar>(y,x) = m_OpenCVImage_.at<uchar>(yIndex, xIndex); 
        }
    }

    return ImageGray(newImage);
}

ImageGray ImageGray::RescaleBilinear(uint newHeight, uint newWidth)
{   
    if (GetHeight() == newHeight && GetWidth() == newWidth)
    {
        return ImageGray(*this);
    }

    cv::Mat newImage(newHeight, newWidth, CV_8UC1);
    double xScaleFactor =  static_cast<double>(GetWidth()) / static_cast<double>(newWidth);
    double yScaleFactor =  static_cast<double>(GetHeight()) / static_cast<double>(newHeight);

    double maxWidthIndexDouble = static_cast<double>(GetWidth() - 1);
    double maxHeightIndexDouble = static_cast<double>(GetHeight() - 1);
    
    for (size_t y = 0; y < newHeight; y++)
    {
        double ySource = std::clamp((static_cast<double>(y) +0.5) * yScaleFactor - 0.5, 0.0, maxHeightIndexDouble);
            
        double yUp = std::floor(ySource);
        double yDown = std::ceil(ySource);

        uint yDownIndex = static_cast<uint>(yDown);
        uint yUpIndex = static_cast<uint>(yUp);

        double yUpToSourceDistance = ySource - yUp;


        for (size_t x = 0; x < newWidth; x++)
        {
            double xSource = std::clamp((static_cast<double>(x) + 0.5) * xScaleFactor - 0.5, 0.0, maxWidthIndexDouble);
                
            double xLeft = std::floor(xSource);
            double xRight = std::ceil(xSource);
                
            uint xLeftIndex = static_cast<uint>(xLeft);
            uint xRightIndex = static_cast<uint>(xRight);
                
            double xLeftToSourceDistance = xSource - xLeft;
            double topLeftGrayValue = static_cast<double>(m_OpenCVImage_.at<uchar>(yUpIndex, xLeftIndex));
            double bottomLeftGrayValue = static_cast<double>(m_OpenCVImage_.at<uchar>(yDownIndex, xLeftIndex));
            double topRightGrayValue = static_cast<double>(m_OpenCVImage_.at<uchar>(yUpIndex, xRightIndex));
            double bottomRightGrayValue = static_cast<double>(m_OpenCVImage_.at<uchar>(yDownIndex, xRightIndex));
                
            double a1 = std::lerp(topLeftGrayValue, topRightGrayValue, xLeftToSourceDistance);
            double a2 = std::lerp(bottomLeftGrayValue, bottomRightGrayValue, xLeftToSourceDistance);
            double a3 = std::lerp(a1, a2, yUpToSourceDistance);
                
            uchar newPixel = static_cast<uchar>(a3);
                
            // Apply pixel value
            newImage.at<uchar>(y,x) = newPixel; 
        }
    }
    
    return ImageGray(newImage);
}

ImageGray ImageGray::RescaleBicubic(uint newHeight, uint newWidth)
{
    if (GetHeight() == newHeight && GetWidth() == newWidth)
    {
        return ImageGray(*this);
    }
    
    cv::Mat newImage(newHeight, newWidth, CV_8UC1);
    double xScaleFactor =  static_cast<double>(GetWidth()) / static_cast<double>(newWidth);
    double yScaleFactor =  static_cast<double>(GetHeight()) / static_cast<double>(newHeight);

    double maxWidthIndexDouble = static_cast<double>(GetWidth() - 1);
    double maxHeightIndexDouble = static_cast<double>(GetHeight() - 1);
    
    for (size_t y = 0; y < newHeight; y++)
    {
        double ySource = std::clamp((static_cast<double>(y) +0.5) * yScaleFactor - 0.5, 0.0, maxHeightIndexDouble);
            
        int yBase = static_cast<int>(std::floor(ySource));
        double yDistance = ySource - static_cast<double>(yBase);

        int y0 = std::clamp(yBase - 1, 0, static_cast<int>(GetHeight()) - 1);
        int y1 = std::clamp(yBase,     0, static_cast<int>(GetHeight()) - 1);
        int y2 = std::clamp(yBase + 1, 0, static_cast<int>(GetHeight()) - 1);
        int y3 = std::clamp(yBase + 2, 0, static_cast<int>(GetHeight()) - 1);

        for (size_t x = 0; x < newWidth; x++)
        {
            double xSource = std::clamp((static_cast<double>(x) + 0.5) * xScaleFactor - 0.5, 0.0, maxWidthIndexDouble);
                
            int xBase = static_cast<int>(std::floor(xSource));
            double xDistance = xSource - static_cast<double>(xBase);

            int x0 = std::clamp(xBase - 1, 0, static_cast<int>(GetWidth()) - 1);
            int x1 = std::clamp(xBase,     0, static_cast<int>(GetWidth()) - 1);
            int x2 = std::clamp(xBase + 1, 0, static_cast<int>(GetWidth()) - 1);
            int x3 = std::clamp(xBase + 2, 0, static_cast<int>(GetWidth()) - 1);
                
            double row0 = CardinalCubic(
                static_cast<double>(m_OpenCVImage_.at<uchar>(y0, x0)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y0, x1)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y0, x2)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y0, x3)),
                xDistance);

            double row1 = CardinalCubic(
                static_cast<double>(m_OpenCVImage_.at<uchar>(y1, x0)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y1, x1)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y1, x2)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y1, x3)),
                xDistance);

            double row2 = CardinalCubic(
                static_cast<double>(m_OpenCVImage_.at<uchar>(y2, x0)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y2, x1)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y2, x2)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y2, x3)),
                xDistance);

            double row3 = CardinalCubic(
                static_cast<double>(m_OpenCVImage_.at<uchar>(y3, x0)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y3, x1)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y3, x2)),
                static_cast<double>(m_OpenCVImage_.at<uchar>(y3, x3)),
                xDistance);

            double newValue = CardinalCubic(row0, row1, row2, row3, yDistance);

            newValue = std::clamp(newValue, 0.0, 255.0);

            newImage.at<uchar>(y, x) = static_cast<uchar>(newValue);                
        }
    }
    
    return ImageGray(newImage);
}

// see https://en.wikipedia.org/wiki/Bicubic_interpolation
// https://www.cs.ubc.ca/~rhodin/2023_2024_CPSC_427/lectures/14_curves_and_animation.pdf?utm_source=chatgpt.com
double ImageGray::CardinalCubic(double p0, double p1, double p2, double p3, double t)
{
    constexpr double tension = 0.5;   // Catmull–Rom
        
    double m1 = tension * (p2 - p0);
    double m2 = tension * (p3 - p1);
        
    double t2 = t * t;
    double t3 = t2 * t;
        
    double h00 =  2.0 * t3 - 3.0 * t2 + 1.0;
    double h10 = t3 - 2.0 * t2 + t;
    double h01 = -2.0 * t3 + 3.0 * t2;
    double h11 = t3 - t2;
        
    return h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2;
};