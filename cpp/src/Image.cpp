#include <Image.hpp>
#include <opencv2/opencv.hpp>
#include <string>

void Image::Display(const std::string windowName)
{
    cv::imshow(windowName, m_OpenCVImage_);
}
