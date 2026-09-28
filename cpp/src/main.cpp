#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <Images/ImageBGR.hpp>
#include <Quantizers/ScalarQuantizer.hpp>
#include <Quantizers/VectorQuantizer.hpp>
#include <Quantizers/DPCMQuantizer.hpp>
#include <Quantizers/BTCQuantizer.hpp>

int main(int argcount, char** args)
{
    ImageBGR image = ImageBGR("/home/math/Documents/school/S8/App2/images/mandrill.tif");
    // image.Display("BGR");
    
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    ImageGray grayImage = image.GetAsGrayImage();
    ImageGray bicubic = grayImage.RescaleBicubic(256, 256);
    
    // Uniform
    uchar* uniformData = UniformQuantizer::QuantizeAndPack(bicubic, 5);
    ImageGray uniformImage = UniformQuantizer::Unpack(uniformData);
    uniformImage.Display("Uniform");
    delete[] uniformData;
    double uniformPSNR = bicubic.CalculatePSNR(uniformImage);
    std::cout << "Uniform PSNR :" << uniformPSNR << std::endl; 

    // Vector
    uchar* vectordata = VectorQuantizer::QuantizeAndPack(bicubic, 5, 1);
    ImageGray vectorImage = VectorQuantizer::Unpack(vectordata);
    vectorImage.Display("Vector");
    delete[] vectordata;
    double vectorPSNR = bicubic.CalculatePSNR(vectorImage);
    std::cout << "Vector PSNR :" << vectorPSNR << std::endl; 

    // Gauss
    uchar* gaussData = GaussQuantizer::QuantizeAndPack(bicubic, 5);
    ImageGray gaussImage = GaussQuantizer::Unpack(gaussData);
    gaussImage.Display("Gauss");
    delete[] gaussData;
    double GaussPSNR = bicubic.CalculatePSNR(gaussImage);
    std::cout << "Guass PSNR :" << GaussPSNR << std::endl; 

    // DPCM 
    uchar* dpcmdata = DPCMQuantizer::QuantizeAndPack(bicubic, 5);
    ImageGray dpcmImage = DPCMQuantizer::Unpack(dpcmdata);
    dpcmImage.Display("DPCM");
    delete[] dpcmdata;
    double DPCMPSNR = bicubic.CalculatePSNR(dpcmImage);
    std::cout << "DPCM PSNR :" << DPCMPSNR << std::endl; 

    // BTC
    uchar* btcdata = BTCQuantizer::QuantizeAndPack(bicubic, 8, 2);
    ImageGray btcImage = BTCQuantizer::Unpack(btcdata);
    btcImage.Display("BTC");
    delete[] btcdata;
    double BTCSNR = bicubic.CalculatePSNR(btcImage);
    std::cout << "BTC PSNR :" << BTCSNR << std::endl;

    // DCT
    uchar* dctData = BTCQuantizer::QuantizeAndPack(bicubic, 8, 2);
    ImageGray dctImage = BTCQuantizer::Unpack(dctData);
    dctImage.Display("DCT");
    delete[] dctData;
    double DCTSNR = bicubic.CalculatePSNR(dctImage);
    std::cout << "DCT PSNR :" << DCTSNR << std::endl; 

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    cv::waitKey();

} 