#pragma once

#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <opencv2/opencv.hpp>

class Image {
public:
    Image() = default;

    inline uint GetWidth() const { return static_cast<uint>(m_OpenCVImage_.cols); }
    inline uint GetHeight() const { return static_cast<uint>(m_OpenCVImage_.rows); }
    inline uint GetChannelCount() const { return static_cast<uint>(m_OpenCVImage_.channels()); }
    
    inline uchar* GetData() const { return m_OpenCVImage_.data; }
    inline const uchar* GetConstData() const { return m_OpenCVImage_.data; }

    cv::Vec3b GetPixel(uint x, uint y) {  return m_OpenCVImage_.at<cv::Vec3b>(y, x);} 
    void Display(const std::string windowName);
protected:
    cv::Mat m_OpenCVImage_;
};