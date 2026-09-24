#include <ImageBGR.hpp>
#include <cassert>
#include <opencv2/opencv.hpp>

ImageBGR::ImageBGR(const std::string &imagePath)
{
    m_OpenCVImage_ = cv::imread(imagePath, cv::IMREAD_COLOR);
}

ImageGray ImageBGR::GetAsGrayImage()
{    
    cv::Mat newImage(GetHeight(), GetWidth(), CV_8UC1);

    // 0.2989 * 16384 ≈ 4898 (Red)
    // 0.5870 * 16384 ≈ 9617 (Green)
    // 0.1140 * 16384 ≈ 1868 (Blue)
    cv::Vec3i ratio = {1868, 9617, 4898};
    for (size_t x = 0; x < static_cast<size_t>(GetWidth()); x++)
    {
        for (size_t y = 0; y < static_cast<size_t>(GetHeight()); y++)
        {
            cv::Vec3i BGR = static_cast<cv::Vec3i>(m_OpenCVImage_.at<cv::Vec3b>(y, x));
            cv::Vec3i newValue = BGR.mul(ratio);
            newImage.at<uchar>(y,x) = static_cast<uchar>((newValue[0] + newValue[1] + newValue[2]) >> 14);
        }
    }

    return ImageGray(newImage);
}
