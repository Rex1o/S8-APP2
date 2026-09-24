#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <ImageBGR.hpp>

int main(int argcount, char** args)
{
    ImageBGR image = ImageBGR("/home/math/Documents/school/S8/App2/images/vitalii-khodzinskyi-lKR2kxzNxzg-unsplash.jpg");
    image.Display("BGR");
    
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    ImageGray grayImage = image.GetAsGrayImage();
    
    // grayImage.Display("Gray");
    
    // ImageGray rescaled = grayImage.RescaleNearestNeigbor(512, 512);
    // rescaled.Display("RescaledNN");
    
    // ImageGray bilinear = grayImage.RescaleBilinear(512, 512);
    // bilinear.Display("Bilinear");
    
    ImageGray bicubic = grayImage.RescaleBicubic(256, 256);
    // bicubic.Display("Bicubic");

    // cv::waitKey();

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
} 