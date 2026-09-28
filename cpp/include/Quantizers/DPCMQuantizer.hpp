#pragma once
#include <Images/ImageGray.hpp>

struct DPCMHeader
{
    uint32_t width;
    uint32_t height;

    float errorMean;
    float errorStdDev;

    uint8_t bitCount;
};

class DPCMQuantizer
{
public:
    static uchar* QuantizeAndPack(const ImageGray& image, uint8_t bitCount, float& bitPerPixel);
    static ImageGray Unpack(uchar* packedData);
    static void CalculateMEDPredictionErrorStatistics(const ImageGray& image, float& outMean, float& outStdDev);
};
