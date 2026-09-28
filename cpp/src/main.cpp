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
#include <Quantizers/DCTQuantizer.hpp>

int main(int argcount, char** args)
{
    ImageBGR image = ImageBGR("/home/math/Documents/school/S8/App2/images/mandrill.tif");
    
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    ImageGray grayImage = image.GetAsGrayImage();
    ImageGray bicubic = grayImage.RescaleBicubic(256, 256);
    
    // Uniform
    float uniformBitsPerPixel = 0.0;
    uchar* uniformData = UniformQuantizer::QuantizeAndPack(bicubic, 5, uniformBitsPerPixel);
    ImageGray uniformImage = UniformQuantizer::Unpack(uniformData);
    uniformImage.Display("Uniform");
    delete[] uniformData;
    double uniformPSNR = bicubic.CalculatePSNR(uniformImage);
    std::cout << "Uniform | PSNR : " << uniformPSNR << " | Bits per pixel : " << uniformBitsPerPixel << std::endl; 

    // Vector
    float vectorBitsPerPixel = 0.0;
    uchar* vectordata = VectorQuantizer::QuantizeAndPack(bicubic, 5, 1, vectorBitsPerPixel);
    ImageGray vectorImage = VectorQuantizer::Unpack(vectordata);
    vectorImage.Display("Vector");
    delete[] vectordata;
    double vectorPSNR = bicubic.CalculatePSNR(vectorImage);
    std::cout << "Vector | PSNR : " << vectorPSNR << " | Bits per pixel : " << vectorBitsPerPixel << std::endl; 

    // Gauss
    float gaussBitsPerPixel = 0.0;
    uchar* gaussData = GaussQuantizer::QuantizeAndPack(bicubic, 5, gaussBitsPerPixel);
    ImageGray gaussImage = GaussQuantizer::Unpack(gaussData);
    gaussImage.Display("Gauss");
    delete[] gaussData;
    double GaussPSNR = bicubic.CalculatePSNR(gaussImage);
    std::cout << "Guass | PSNR : " << GaussPSNR << " | Bits per pixel : " << gaussBitsPerPixel << std::endl; 

    // DPCM
    float dpcmBitsPerPixel = 0.0;
    uchar* dpcmdata = DPCMQuantizer::QuantizeAndPack(bicubic, 5, dpcmBitsPerPixel);
    ImageGray dpcmImage = DPCMQuantizer::Unpack(dpcmdata);
    dpcmImage.Display("DPCM");
    delete[] dpcmdata;
    double DPCMPSNR = bicubic.CalculatePSNR(dpcmImage);
    std::cout << "DPCM | PSNR : " << DPCMPSNR << " | Bits per pixel : " << dpcmBitsPerPixel << std::endl; 

    // BTC
    float btcBitsPerPixel = 0.0;
    uchar* btcdata = BTCQuantizer::QuantizeAndPack(bicubic, 8, 2, btcBitsPerPixel);
    ImageGray btcImage = BTCQuantizer::Unpack(btcdata);
    btcImage.Display("BTC");
    delete[] btcdata;
    double BTCSNR = bicubic.CalculatePSNR(btcImage);
    std::cout << "BTC | PSNR : " << BTCSNR << " | Bits per pixel : " << btcBitsPerPixel << std::endl;

    // DCT
    float dctBitsPerPixel = 0.0;
    uchar* dctData = DCTQuantizer::QuantizeAndPack(bicubic, 5, 2, dctBitsPerPixel);
    ImageGray dctImage = DCTQuantizer::Unpack(dctData);
    dctImage.Display("DCT");
    delete[] dctData;
    double DCTSNR = bicubic.CalculatePSNR(dctImage);
    std::cout << "DCT | PSNR : " << DCTSNR << " | Bits per pixel : " << dctBitsPerPixel << std::endl; 

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    cv::waitKey();
} 