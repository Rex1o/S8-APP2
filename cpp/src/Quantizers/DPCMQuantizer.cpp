#include <Quantizers/DPCMQuantizer.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <Quantizers/GaussianPixelQuantizer.hpp>
#include <BitStream.hpp>

// Median Edge Detector Prediction
static int PredictMED(const cv::Mat& reconstructedImage, uint32_t x, uint32_t y)
{
    // -------------------
    // |        |        |
    // |    A   |    B   |
    // -------------------
    // |        |        |
    // |    C   |    X   |
    // -------------------

    // No neighbor for first pixel
    if (x == 0 && y == 0)
    {
        return 0;
    }

    // First row: only left neighbor exists.
    if (y == 0)
    {
        return static_cast<int>(reconstructedImage.at<uchar>(y, x - 1));
    }

    // First column: only top neighbor exists.
    if (x == 0)
    {
        return static_cast<int>(reconstructedImage.at<uchar>(y - 1, x));
    }

    const int A = static_cast<int>(reconstructedImage.at<uchar>(y - 1, x - 1));
    const int B = static_cast<int>(reconstructedImage.at<uchar>(y - 1, x));
    const int C = static_cast<int>(reconstructedImage.at<uchar>(y, x - 1));

    // Median edge detector predictor
    //    | min(B, C) if A >= max(B,C)
    // X' | max(B, C) if A <= min(B,C)
    //    | B + C - A    otherwise
    
    if (A >= std::max(B, C))
    {
        return std::min(B, C);
    }

    if (A <= std::min(B, C))
    {
        return std::max(B, C);
    }

    return B + C - A;
}

void DPCMQuantizer::CalculateMEDPredictionErrorStatistics(const ImageGray& image, float& outMean, float& outStdDev)
{
    const uint32_t width = image.GetWidth();
    const uint32_t height = image.GetHeight();

    double sum = 0.0;
    double squaredSum = 0.0;
    size_t count = 0;

    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            // First pixel is transmitted directly.
            if (x == 0 && y == 0)
            {
                continue;
            }
            
            const int pixel = static_cast<int>(image.m_OpenCVImage_.at<uchar>(y, x));
            int prediction = PredictMED(image.m_OpenCVImage_, x ,y);

            const int error = pixel - prediction;

            sum += static_cast<double>(error);
            squaredSum += static_cast<double>(error) * static_cast<double>(error);
            ++count;
        }
    }

    if (count == 0)
    {
        outMean = 0.0f;
        outStdDev = 0.0f;
        return;
    }

    const double mean = sum / static_cast<double>(count);
    const double variance = squaredSum / static_cast<double>(count) - mean * mean;
    outMean = static_cast<float>(mean);
    outStdDev = static_cast<float>(std::sqrt(std::max(0.0, variance)));
}


PackedData DPCMQuantizer::QuantizeAndPack(const ImageGray& image,uint8_t bitCount)
{
    const uint32_t width = image.GetWidth();
    const uint32_t height = image.GetHeight();
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);

    float errorMean;
    float errorStdDev;

    CalculateMEDPredictionErrorStatistics(image, errorMean, errorStdDev);

    GaussianErrorQuantizer gaussQuantizer(bitCount, errorMean, errorStdDev);

    // First pixel is it's full = 8 bits
    // Remaining pixels = bitCount bits each
    const size_t dataBitCount = 8 + (pixelCount - 1) * bitCount;
    const size_t packedDataSize = (dataBitCount + 7) / 8;

    const size_t outputSize = sizeof(DPCMHeader) + packedDataSize;
    
    uchar* output = new uchar[outputSize]{};

    DPCMHeader header;
    header.width = width;
    header.height = height;
    header.errorMean = errorMean;
    header.errorStdDev = errorStdDev;
    header.bitCount = bitCount;

    std::memcpy(output, &header, sizeof(DPCMHeader));

    uchar* data = output + sizeof(DPCMHeader);
    BitWriter writer(data);

    cv::Mat reconstructedImage(height, width, CV_8UC1);

    const uchar firstPixel = image.GetPixel(0, 0);
 
    writer.Write(firstPixel, 8);

    reconstructedImage.at<uchar>(0, 0) = firstPixel;

    // ------------------------------------------------------------------------
    // Actual closed-loop MED DPCM encoding.
    // ------------------------------------------------------------------------

    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            if (x == 0 && y == 0)
            {
                continue;
            }

            // Original pixel.
            const int pixel = static_cast<int>(image.m_OpenCVImage_.at<uchar>(y, x));

            // MED prediction using RECONSTRUCTED neighbors.
            const int prediction = PredictMED(reconstructedImage, x, y);

            // Calculate Error
            const int error = pixel - prediction;
            
            // Quatize error
            const uint8_t quantizedError = gaussQuantizer.Quantize(error);
            writer.Write(quantizedError, bitCount);

            // We must use the reconstructed pixel to predict the next one, so save it in a new image.
            const int reconstructedError = gaussQuantizer.Dequantize(quantizedError);
            const int reconstructedPixel = std::clamp(prediction + reconstructedError, 0,255);
            reconstructedImage.at<uchar>(y, x) = static_cast<uchar>(reconstructedPixel);
        }
    }

    writer.Flush();

    float bitPerPixel = static_cast<float>(outputSize) / static_cast<float>(pixelCount) * 8.0f;
    return PackedData {output, outputSize, bitPerPixel};
}

ImageGray DPCMQuantizer::Unpack(uchar* packedData)
{
    DPCMHeader header;

    std::memcpy(&header, packedData, sizeof(DPCMHeader));

    GaussianErrorQuantizer gaussQuantizer(header.bitCount, header.errorMean, header.errorStdDev);

    BitReader reader(packedData + sizeof(DPCMHeader));

    cv::Mat reconstructedImage(header.height, header.width, CV_8UC1);

    const uchar firstPixel = reader.Read(8);

    reconstructedImage.at<uchar>(0, 0) = firstPixel;

    for (uint32_t y = 0; y < header.height; ++y)
    {
        for (uint32_t x = 0; x < header.width; ++x)
        {
            if (x == 0 && y == 0)
            {
                continue;
            }

            const int prediction = PredictMED(reconstructedImage, x, y);

            // Read Gaussian quantizer index.
            const uint8_t quantizedError = reader.Read(header.bitCount);

            // Convert index back into SIGNED error.
            const int reconstructedError = gaussQuantizer.Dequantize(quantizedError);

            // Reconstruct pixel.
            const int reconstructedPixel = std::clamp(prediction + reconstructedError, 0, 255);

            reconstructedImage.at<uchar>(y, x) = static_cast<uchar>(reconstructedPixel);
        }
    }

    return ImageGray(reconstructedImage);
}