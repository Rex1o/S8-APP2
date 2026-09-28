#pragma once
#include <Images/ImageGray.hpp>

struct UniformImageHeader
{
    uint32_t width;
    uint32_t height;
    uint8_t bitsPerPixel;
};

class UniformQuantizer
{
public:
    static uchar* QuantizeAndPack(const ImageGray& image, uint8_t bitCount);
    static ImageGray Unpack(uchar* packedData);
};

struct GaussImageHeader
{
    uint32_t width;
    uint32_t height;
    float mean;
    float stdDev;
    uint8_t bitsPerPixel;
};

class GaussQuantizer
{
public:
    static uchar* QuantizeAndPack(const ImageGray& image, uint8_t bitCount);
    static ImageGray Unpack(uchar* packedData);
};

struct JayantQuantizedImageHeader
{
    uint32_t width;
    uint32_t height;
    float initialDelta;
    uint8_t bitsPerPixel;
};

// TODO comme back here, image is not pretty at all.
class JayantQuantizer
{
public:
    static uchar* QuantizeAndPack(const ImageGray& image, uint8_t bitCount);
    static ImageGray Unpack(uchar* packedData);

private:
    static uint32_t Quantize(float value, float delta, uint8_t bitCount);
    static float Reconstruct(uint32_t index, float delta, uint8_t bitCount);
    static float GetMultiplier(uint32_t index, uint8_t bitCount);
};


