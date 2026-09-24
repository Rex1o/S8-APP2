#pragma once
#include <Image.hpp>
#include <ImageGray.hpp>
#include <string>

class ImageBGR : public Image
{
public:
    ImageBGR(const std::string& imagePath);
    cv::Vec3b GetPixel(uint x, uint y)
    {
        assert(m_OpenCVImage_.type() == CV_8UC1);
        return m_OpenCVImage_.at<cv::Vec3b>(y, x);
    } 
    ImageGray GetAsGrayImage();
private:

};