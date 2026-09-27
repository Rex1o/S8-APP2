#pragma once
#include <Images/ImageGray.hpp>

class DPCMQuantizer
{
public:
    static uchar* QuantizeAndPack(const ImageGray& image, uint8_t bitCount);
    static ImageGray Unpack(uchar* packedData);
};
