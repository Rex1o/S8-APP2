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
    static uchar* QuantizeAndPack(const ImageGray &image, uint8_t bitCount, uint8_t blocSize);
    static ImageGray Unpack(uchar* packedData);
};
