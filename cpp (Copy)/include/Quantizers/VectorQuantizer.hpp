#pragma once
#include <Images/ImageGray.hpp>

struct VectorQuantizedImageHeader
{
    uint32_t width;
    uint32_t height;
    uint8_t vectorBitCount;
    uint8_t blocSize;
};

struct VectorArray
{
    uint8_t vectorCount;
    uint8_t vectorSize;
    double* values;
    
    VectorArray(uint8_t vectorCount, uint8_t vectorSize);
    
    ~VectorArray();

    VectorArray(const VectorArray&) = delete;
    VectorArray& operator=(const VectorArray&) = delete;

    VectorArray(VectorArray&& other) noexcept;
    VectorArray& operator=(VectorArray&& other) noexcept;
    
    double* operator[](uint32_t vectorIndex);
    const double* operator[](uint32_t vectorIndex) const;
    VectorArray DoubleVectors();
};

struct UcharDict
{
    uint8_t vectorCount;
    uint8_t vectorSize;
    uchar* values;

    UcharDict(uint8_t vectorCount, uint8_t vectorSize);
    UcharDict(const VectorArray& vectorArray);

    UcharDict(uchar* data);

    ~UcharDict();

    UcharDict(const UcharDict&) = delete;
    UcharDict& operator=(const UcharDict&) = delete;

    UcharDict(UcharDict&& other) noexcept;

    UcharDict& operator=(UcharDict&& other) noexcept;

    uchar* operator[](uint32_t vectorIndex);

    const uchar* operator[](uint32_t vectorIndex) const;

    size_t GetDataSize() const;
};



class VectorQuantizer
{
public:
    static uchar* QuantizeAndPack(const ImageGray& image, uint8_t vectorBitCount, uint8_t blocSize);
    static ImageGray Unpack(uchar* packedData);

private:
    static VectorArray TrainLBG(const ImageGray &image, uint8_t vectorBitCount, uint8_t blocSize);
    static double RecalculateCentroids(const ImageGray& image, VectorArray& vectorArray, uint8_t blocSize);
    static uchar* FindBlockIndices(const ImageGray& image,const VectorArray& vectorArray, uint8_t blocSize, uint32_t& blockCount);
};
