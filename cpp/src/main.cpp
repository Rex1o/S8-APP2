#include <chrono>
#include <filesystem>
#include <fstream>
#include <Images/ImageBGR.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Quantizers/BTCQuantizer.hpp>
#include <Quantizers/DCTQuantizer.hpp>
#include <Quantizers/DPCMQuantizer.hpp>
#include <Quantizers/ScalarQuantizer.hpp>
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

    // mandrill.tif -> mandrill/
    fs::path outputDirectory = imagePath.parent_path() / imagePath.stem();

    fs::create_directories(outputDirectory);

    ImageBGR image(imagePath.string());

    auto start = std::chrono::high_resolution_clock::now();

    ImageGray grayImage = image.GetAsGrayImage();
    ImageGray bicubic = grayImage.RescaleBicubic(256, 256);

    cv::imwrite((outputDirectory / "original.png").string(), bicubic.m_OpenCVImage_);

    // Uniform
    PackedData uniformData = UniformQuantizer::QuantizeAndPack(bicubic, 5);
    fs::path uniformPath = outputDirectory / "uniform.bin";
    WritePackedData(uniformPath, uniformData);
    // Delete memory to guarantee that Unpack isn't using it.
    delete[] uniformData.data;
    uniformData.data = nullptr;
    std::vector<uchar> uniformFileData = ReadPackedData(uniformPath);
    ImageGray uniformImage = UniformQuantizer::Unpack(uniformFileData.data());
    cv::imwrite((outputDirectory / "uniform.png").string(), uniformImage.m_OpenCVImage_);
    double uniformPSNR = bicubic.CalculatePSNR(uniformImage);
    std::cout << "Uniform | PSNR : " << uniformPSNR << " | Bits per pixel : " << uniformData.bitsPerPixel << std::endl;

    // Vector
    PackedData vectorData = VectorQuantizer::QuantizeAndPack(bicubic, 5, 1);
    fs::path vectorPath = outputDirectory / "vector.bin";
    WritePackedData(vectorPath, vectorData);
    delete[] vectorData.data;
    vectorData.data = nullptr;
    std::vector<uchar> vectorFileData = ReadPackedData(vectorPath);
    ImageGray vectorImage = VectorQuantizer::Unpack(vectorFileData.data());
    cv::imwrite((outputDirectory / "vector.png").string(), vectorImage.m_OpenCVImage_);
    double vectorPSNR = bicubic.CalculatePSNR(vectorImage);
    std::cout << "Vector | PSNR : " << vectorPSNR << " | Bits per pixel : " << vectorData.bitsPerPixel << '\n';

    // Gaussian
    PackedData gaussData = GaussQuantizer::QuantizeAndPack(bicubic, 5);
    fs::path gaussPath = outputDirectory / "gauss.bin";
    WritePackedData(gaussPath, gaussData);
    delete[] gaussData.data;
    gaussData.data = nullptr;
    std::vector<uchar> gaussFileData = ReadPackedData(gaussPath);
    ImageGray gaussImage = GaussQuantizer::Unpack(gaussFileData.data());
    cv::imwrite((outputDirectory / "gauss.png").string(),gaussImage.m_OpenCVImage_);
    double gaussPSNR = bicubic.CalculatePSNR(gaussImage);
    std::cout << "Gauss | PSNR : " << gaussPSNR << " | Bits per pixel : " << gaussData.bitsPerPixel << std::endl;

    // DPCM
    PackedData dpcmData = DPCMQuantizer::QuantizeAndPack(bicubic, 5);
    fs::path dpcmPath = outputDirectory / "dpcm.bin";
    WritePackedData(dpcmPath, dpcmData);
    delete[] dpcmData.data;
    dpcmData.data = nullptr;
    std::vector<uchar> dpcmFileData = ReadPackedData(dpcmPath);
    ImageGray dpcmImage = DPCMQuantizer::Unpack(dpcmFileData.data());
    cv::imwrite((outputDirectory / "dpcm.png").string(),dpcmImage.m_OpenCVImage_);
    double dpcmPSNR = bicubic.CalculatePSNR(dpcmImage);
    std::cout << "DPCM | PSNR : " << dpcmPSNR << " | Bits per pixel : " << dpcmData.bitsPerPixel << std::endl;

    // BTC
    PackedData btcData = BTCQuantizer::QuantizeAndPack(bicubic, 8, 2);
    fs::path btcPath = outputDirectory / "btc.bin";
    WritePackedData(btcPath, btcData);
    delete[] btcData.data;
    btcData.data = nullptr;
    std::vector<uchar> btcFileData = ReadPackedData(btcPath);
    ImageGray btcImage = BTCQuantizer::Unpack(btcFileData.data());
    cv::imwrite((outputDirectory / "btc.png").string(),btcImage.m_OpenCVImage_);
    double btcPSNR = bicubic.CalculatePSNR(btcImage);

    std::cout << "BTC | PSNR : " << btcPSNR << " | Bits per pixel : " << btcData.bitsPerPixel << std::endl;

    // DCT
    PackedData dctData = DCTQuantizer::QuantizeAndPack(bicubic, 5, 2);
    fs::path dctPath = outputDirectory / "dct.bin";
    WritePackedData(dctPath, dctData);
    delete[] dctData.data;
    dctData.data = nullptr;
    std::vector<uchar> dctFileData = ReadPackedData(dctPath);
    ImageGray dctImage = DCTQuantizer::Unpack(dctFileData.data());
    cv::imwrite((outputDirectory / "dct.png").string(),dctImage.m_OpenCVImage_);
    double dctPSNR = bicubic.CalculatePSNR(dctImage);
    std::cout << "DCT | PSNR : " << dctPSNR << " | Bits per pixel : " << dctData.bitsPerPixel << std::endl;

    // End clock
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\nTime: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    std::cout << "Results saved in: " << outputDirectory << '\n';

    return 0;
}