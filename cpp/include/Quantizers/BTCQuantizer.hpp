#pragma once
#include <Images/ImageGray.hpp>

struct BTCHeader
{
    uint32_t width;
    uint32_t height;
    uint8_t blocSize;
    uint8_t resconstructLevelBitCount;
};

// |Header|A1|B1|bloc2|A2|B2|bloc2
class BTCQuantizer
{
public:
    static uchar* QuantizeAndPack(const ImageGray& image, uint8_t resconstructLevelBitCount, uint8_t blocSize, float& bitPerPixel);
    static ImageGray Unpack(uchar* packedData);
    static void MinimizeMSE(cv::Mat bloc, double& A, double& B, uchar& outSplitValue, size_t& outEqualNeeded);
};