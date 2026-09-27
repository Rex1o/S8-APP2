#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <Images/ImageBGR.hpp>
#include <Quantizers/ScalarQuantizer.hpp>
#include <Quantizers/VectorQuantizer.hpp>

int main(int argcount, char** args)
{
    ImageBGR image = ImageBGR("/home/math/Documents/school/S8/App2/images/tulips.png");
    // image.Display("BGR");
    
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    ImageGray grayImage = image.GetAsGrayImage();

    ImageGray bicubic = grayImage.RescaleBicubic(256, 256);
    ImageGray bilinear = grayImage.RescaleBilinear(256, 256);
    bicubic.Display("Bicubic");
    bilinear.Display("Bilinear");

    uchar* data = JayantQuantizer::QuantizeAndPack(bicubic, 3);
    ImageGray vectorImage =  JayantQuantizer::Unpack(data);
    vectorImage.Display("Jayant");

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    cv::waitKey();

} 