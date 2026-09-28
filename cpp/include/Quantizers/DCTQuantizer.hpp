#pragma once
#include <Images/ImageGray.hpp>

struct DCTHeader
{
    uint32_t width;
    uint32_t height;
    uint8_t blocSize;
};

class DCTQuantizer
{
public:
    static uchar* QuantizeAndPack(const ImageGray &image, uint8_t bitCount, uint8_t blocSize, float& bitPerPixel);
    static ImageGray Unpack(uchar* packedData);
};
