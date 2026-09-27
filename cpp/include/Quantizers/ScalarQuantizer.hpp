#pragma once
#include <Images/ImageGray.hpp>

struct ScalarQuantizedImageHeader
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

struct AdaptiveGaussQuantizedImageHeader
{
    uint32_t width;
    uint32_t height;
    float mean;
    float variance;
    uint8_t bitsPerPixel;
};

// This implemenation does not do a block per block version.
// This is a guass representation of the entire image.
// We could in the future make it so we add a headers for group of pixels. Let's say 8*8 pixels and then add the header per bloc of 8 bits for the max
// and min values. 
class FowardAdaptiveGaussQuantizer
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


