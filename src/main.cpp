#include <chrono>
#include <filesystem>
#include <fstream>
#include <Images/ImageBGR.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Quantizers/BTCQuantizer.hpp>
#include <Quantizers/DCTQuantizer.hpp>
#include <Quantizers/DPCMQuantizer.hpp>
#include <Quantizers/VectorQuantizer.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void WritePackedData(const fs::path& path, const PackedData& packedData)
{
    std::ofstream file(path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    file.write(reinterpret_cast<const char*>(packedData.data), static_cast<std::streamsize>(packedData.size));
}

std::vector<uchar> ReadPackedData(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file)
    {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uchar> data(static_cast<size_t>(size));

    file.read(reinterpret_cast<char*>(data.data()), size);

    return data;
}

int main(int argcount, char** args)
{
    if (argcount != 2)
    {
        std::cerr << "Usage: " << args[0] << " <image_path>\n";
        return 1;
    }

    fs::path imagePath = args[1];

    if (!fs::exists(imagePath))
    {
        std::cerr << "Image does not exist: " << imagePath << '\n';
        return 1;
    }
    
    // Build all paths
    fs::path outputDirectory = imagePath.parent_path() / imagePath.stem();
    fs::create_directories(outputDirectory);
    fs::path uniformPath = outputDirectory / "uniform.bin";
    fs::path vectorPath = outputDirectory / "vector.bin";
    fs::path gaussPath = outputDirectory / "gauss.bin";
    fs::path dpcmPath = outputDirectory / "dpcm.bin";
    fs::path btcPath = outputDirectory / "btc.bin";
    fs::path dctPath = outputDirectory / "dct.bin";

    ImageBGR I_source(imagePath.string());
    auto start = std::chrono::high_resolution_clock::now();

    ImageGray I_Gray = I_source.GetAsGrayImage();
    ImageGray I_reduced = I_Gray.RescaleBicubic(256, 256);
    cv::imwrite((outputDirectory / "original.png").string(), I_reduced.m_OpenCVImage_);
    
    // Vector
    PackedData I_encodedVector = VectorQuantizer::QuantizeAndPack(I_reduced, 5, 1);
    WritePackedData(vectorPath, I_encodedVector);
    std::vector<uchar> vectorFileData = ReadPackedData(vectorPath);
    ImageGray I_decodedVector = VectorQuantizer::Unpack(vectorFileData.data());

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\nVecteur Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    // DPCM
    PackedData I_encodedDPCM = DPCMQuantizer::QuantizeAndPack(I_reduced, 5);
    WritePackedData(dpcmPath, I_encodedDPCM);
    std::vector<uchar> dpcmFileData = ReadPackedData(dpcmPath);
    ImageGray I_decodedDPCM = DPCMQuantizer::Unpack(dpcmFileData.data());
    
    end = std::chrono::high_resolution_clock::now();
    std::cout << "\nDPCM Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    // BTC
    PackedData I_encodedBTC = BTCQuantizer::QuantizeAndPack(I_reduced, 8, 2);
    WritePackedData(btcPath, I_encodedBTC);
    std::vector<uchar> btcFileData = ReadPackedData(btcPath);
    ImageGray I_decodedBTC = BTCQuantizer::Unpack(btcFileData.data());

    end = std::chrono::high_resolution_clock::now();
    std::cout << "\nBTC Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    // DCT
    // a 4 bit per pixel
    PackedData I_encodedDCT = DCTQuantizer::QuantizeAndPack(I_reduced, 5, 4);
    WritePackedData(dctPath, I_encodedDCT);
    std::vector<uchar> dctFileData = ReadPackedData(dctPath);
    ImageGray I_decodedDCT = DCTQuantizer::Unpack(dctFileData.data());

    end = std::chrono::high_resolution_clock::now();
    std::cout << "\nDCT Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    
    // Release memory
    delete[] I_encodedBTC.data;
    delete[] I_encodedDCT.data;
    delete[] I_encodedDPCM.data;
    delete[] I_encodedVector.data;

    // Write all images
    cv::imwrite((outputDirectory / "btc.png").string(), I_decodedBTC.m_OpenCVImage_);
    cv::imwrite((outputDirectory / "dct.png").string(), I_decodedDCT.m_OpenCVImage_);
    cv::imwrite((outputDirectory / "dpcm.png").string(), I_decodedDPCM.m_OpenCVImage_);
    cv::imwrite((outputDirectory / "vector.png").string(), I_decodedVector.m_OpenCVImage_);
    
    // Calculate PSNR
    double btcPSNR = I_reduced.CalculatePSNR(I_decodedBTC);
    double dctPSNR = I_reduced.CalculatePSNR(I_decodedDCT);
    double dpcmPSNR = I_reduced.CalculatePSNR(I_decodedDPCM);
    double vectorPSNR = I_reduced.CalculatePSNR(I_decodedVector);

    // Display data in console
    std::cout << "BTC | PSNR : " << btcPSNR << " | Bits per pixel : " << I_encodedBTC.bitsPerPixel << std::endl;
    std::cout << "DCT | PSNR : " << dctPSNR << " | Bits per pixel : " << I_encodedDCT.bitsPerPixel << std::endl;
    std::cout << "DPCM | PSNR : " << dpcmPSNR << " | Bits per pixel : " << I_encodedDPCM.bitsPerPixel << std::endl;
    std::cout << "Vector | PSNR : " << vectorPSNR << " | Bits per pixel : " << I_encodedVector.bitsPerPixel << std::endl;
    
    // End clock

    std::cout << "Results saved in: " << outputDirectory << '\n';

    return 0;
}