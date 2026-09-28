#include <Quantizers/ScalarQuantizer.hpp>
#include <BitStream.hpp>
#include <Quantizers/GaussianPixelQuantizer.hpp>
#include <Quantizers/UniformQuantizer.hpp>

// We should try with a Floyd–Steinberg dithering seems like we could have a way better image
PackedData UniformQuantizer::QuantizeAndPack(const ImageGray& image, uint8_t bitCount)
{
    if (bitCount == 0 || bitCount > 8)
    {
        throw std::invalid_argument("M must be between 1 and 8.");
    }

    const size_t pixelCount = image.GetWidth() * image.GetHeight();
    const size_t dataBits = pixelCount * bitCount;
    const size_t packedDataSize = (dataBits + 7) / 8;

    size_t outputSize = sizeof(UniformImageHeader) + packedDataSize;
    unsigned char* output = new unsigned char[outputSize];
    
    UniformImageHeader header;
    header.width = image.GetWidth();
    header.height = image.GetHeight();
    header.bitsPerPixel = bitCount;

    std::memcpy(output, &header, sizeof(header));

    // Packed image starts after header.
    unsigned char* packedData = output + sizeof(header);

    BitWriter writer(packedData);
    UniformPixelQuantizer uniformPixelQuantizer(bitCount);

    for (uint y = 0; y < image.GetHeight(); ++y)
    {
        for (uint x = 0; x < image.GetWidth(); ++x)
        {
            uchar pixel = image.m_OpenCVImage_.at<uchar>(y, x);
            uchar quantized = uniformPixelQuantizer.Quantize(pixel);
            writer.Write(quantized, bitCount);
        }
    }
    writer.Flush();

    float bitPerPixel = static_cast<float>(outputSize) / static_cast<float>(pixelCount) * 8.0f;
    return PackedData {output, outputSize, bitPerPixel};
}

ImageGray UniformQuantizer::Unpack(uchar* packedData)
{
    // Header
    UniformImageHeader header;
    std::memcpy(&header, packedData, sizeof(header));
    uint32_t width = header.width;
    uint32_t height = header.height;
    uchar bitCount = header.bitsPerPixel;
    // Data
    uchar* pixels = packedData + sizeof(UniformImageHeader);

    cv::Mat newImage(height, width, CV_8UC1);    
    UniformPixelQuantizer uniformPixelQuantizer(bitCount);
    
    BitReader reader(pixels);
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            uint32_t quantized = reader.Read(header.bitsPerPixel);
            uchar dequantize = uniformPixelQuantizer.Dequantize(quantized);
            newImage.at<uchar>(y,x) = dequantize; 
        }
    }

    return ImageGray(newImage);
}

PackedData GaussQuantizer::QuantizeAndPack(const ImageGray& image, uint8_t bitCount)
{
    const size_t pixelCount = image.GetWidth() * image.GetHeight();
    const size_t dataBits = pixelCount * bitCount;
    const size_t packedDataSize = (dataBits + 7) / 8;

    float mean;
    float stdDev;

    image.GetMeanAndSTDDev(mean, stdDev);

    const size_t outputSize = sizeof(GaussImageHeader) + packedDataSize;
    uchar* output = new uchar[outputSize];

    GaussImageHeader header;
    header.width = image.GetWidth();
    header.height = image.GetHeight();
    header.mean = mean;
    header.stdDev = stdDev;
    header.bitsPerPixel = bitCount;


    std::memcpy(output, &header, sizeof(header));
    uchar* packedData = output + sizeof(header);
    std::memset(packedData, 0, packedDataSize);
    
    GaussianPixelQuantizer gaussianPixelQuantizer(bitCount, mean, stdDev);
    BitWriter writer(packedData);

    for (uint32_t y = 0; y < image.GetHeight(); ++y)
    {
        for (uint32_t x = 0; x < image.GetWidth(); ++x)
        {
            uchar pixel = image.m_OpenCVImage_.at<uchar>(y, x);
            uchar quantized = gaussianPixelQuantizer.Quantize(pixel);
            writer.Write(quantized, bitCount);
        }
    }

    float bitPerPixel = static_cast<float>(outputSize) / static_cast<float>(pixelCount) * 8.0f;
    return PackedData{output, outputSize, bitPerPixel};
}

ImageGray GaussQuantizer::Unpack(uchar* packedData)
{
    GaussImageHeader header;
    std::memcpy(&header, packedData, sizeof(header));

    cv::Mat image(header.height, header.width, CV_8UC1);
    uchar* compressedData = packedData + sizeof(GaussImageHeader);

    GaussianPixelQuantizer gaussianPixelQuantizer(header.bitsPerPixel, header.mean, header.stdDev);
    BitReader reader(compressedData);

    for (uint32_t y = 0; y < header.height; ++y)
    {
        for (uint32_t x = 0; x < header.width; ++x)
        {
            uchar quantized = reader.Read(header.bitsPerPixel);
            uchar dequantized = gaussianPixelQuantizer.Dequantize(quantized);
            image.at<uchar>(y, x) = cv::saturate_cast<uchar>(std::round(dequantized));
        }
    }

    return ImageGray(image);
}

PackedData JayantQuantizer::QuantizeAndPack(const ImageGray &image, uint8_t bitCount)
{
    if (bitCount != 3)
    {
        throw std::invalid_argument("This implementation currently uses the paper's 3-bit Jayant quantizer.");
    }

    const size_t pixelCount = static_cast<size_t>(image.GetWidth()) * image.GetHeight();
    const size_t dataBits = pixelCount * bitCount;
    const size_t packedDataSize = (dataBits + 7) / 8;
    const size_t outputSize = sizeof(JayantQuantizedImageHeader) + packedDataSize;
    uchar* output = new uchar[outputSize];
    
    JayantQuantizedImageHeader header;
    header.width = image.GetWidth();
    header.height = image.GetHeight();
    header.bitsPerPixel = bitCount;

    // Initial step size.
    header.initialDelta = 32.0f;

    std::memcpy(output, &header, sizeof(header));
    uchar* outputData = output + sizeof(JayantQuantizedImageHeader);
    std::memset(outputData, 0, packedDataSize);
    float delta = header.initialDelta;

    constexpr float MIN_DELTA = 1.0f;
    constexpr float MAX_DELTA = 128.0f;

    uint64_t bitBuffer = 0;
    uint8_t bitsInBuffer = 0;
    size_t outputIndex = 0;

    for (uint32_t y = 0; y < image.GetHeight(); ++y)
    {
        for (uint32_t x = 0; x < image.GetWidth(); ++x)
        {
            float value = static_cast<float>(image.m_OpenCVImage_.at<uchar>(y, x)) - 128.0f;

            uint32_t quantized = Quantize(value, delta, bitCount);

            bitBuffer |= static_cast<uint64_t>(quantized) << bitsInBuffer;

            bitsInBuffer += bitCount;

            while (bitsInBuffer >= 8)
            {
                outputData[outputIndex++] = static_cast<uchar>(bitBuffer & 0xFF);

                bitBuffer >>= 8;
                bitsInBuffer -= 8;
            }

            delta *= GetMultiplier(quantized, bitCount);

            delta = std::clamp(delta, MIN_DELTA, MAX_DELTA);
        }
    }

    if (bitsInBuffer > 0)
    {
        outputData[outputIndex] = static_cast<uchar>(bitBuffer & 0xFF);
    }

    float bitPerPixel = static_cast<float>(outputSize) / static_cast<float>(pixelCount) * 8.0f;
    return PackedData{output, outputSize, bitPerPixel};
}

ImageGray JayantQuantizer::Unpack(uchar *packedData)
{
    JayantQuantizedImageHeader header;
    std::memcpy(&header, packedData, sizeof(JayantQuantizedImageHeader));

    if (header.bitsPerPixel != 3)
    {
        throw std::runtime_error("Unsupported Jayant bit count.");
    }

    cv::Mat image(header.height, header.width, CV_8UC1);

    uchar* inputData = packedData + sizeof(JayantQuantizedImageHeader);

    float delta = header.initialDelta;

    constexpr float MIN_DELTA = 1.0f;
    constexpr float MAX_DELTA = 128.0f;

    uint64_t bitBuffer = 0;
    uint8_t bitsInBuffer = 0;
    size_t inputIndex = 0;

    const uint64_t mask = (1ULL << header.bitsPerPixel) - 1ULL;

    for (uint32_t y = 0; y < header.height; ++y)
    {
        for (uint32_t x = 0; x < header.width; ++x)
        {
            while (bitsInBuffer < header.bitsPerPixel)
            {
                bitBuffer |= static_cast<uint64_t>(inputData[inputIndex++]) << bitsInBuffer;

                bitsInBuffer += 8;
            }

            uint32_t quantized = static_cast<uint32_t>(bitBuffer & mask);

            bitBuffer >>= header.bitsPerPixel;
            bitsInBuffer -= header.bitsPerPixel;

            float reconstructed = Reconstruct(quantized, delta, header.bitsPerPixel);

            reconstructed += 128.0f;

            image.at<uchar>(y, x) = cv::saturate_cast<uchar>(std::round(reconstructed));

            delta *= GetMultiplier(quantized, header.bitsPerPixel);
            delta = std::clamp(delta, MIN_DELTA, MAX_DELTA);
        }
    }

    return ImageGray(image);
}

uint32_t JayantQuantizer::Quantize(float value, float delta, uint8_t bitCount)
{
    const uint32_t levelCount = 1u << bitCount;
    const uint32_t levelsPerSide = levelCount / 2;

    const bool negative = value < 0.0f;
    const float magnitude = std::abs(value);

    uint32_t magnitudeIndex = static_cast<uint32_t>(magnitude / delta);

    // The outermost interval handles overload.
    magnitudeIndex = std::min(magnitudeIndex, levelsPerSide - 1);

    if (negative)
    {
        return levelsPerSide + magnitudeIndex;
    }

    return magnitudeIndex;
}

float JayantQuantizer::Reconstruct(uint32_t index, float delta, uint8_t bitCount)
{
    const uint32_t levelCount = 1u << bitCount;
    const uint32_t levelsPerSide = levelCount / 2;

    const bool negative = index >= levelsPerSide;

    const uint32_t magnitudeIndex = negative ? index - levelsPerSide : index;

    const float magnitude = (static_cast<float>(magnitudeIndex) + 0.5f) * delta;

    return negative ? -magnitude : magnitude;
}

float JayantQuantizer::GetMultiplier(uint32_t index, uint8_t bitCount)
{
    const uint32_t levelCount = 1u << bitCount;
    const uint32_t levelsPerSide = levelCount / 2;

    const uint32_t magnitudeIndex = index >= levelsPerSide ? index - levelsPerSide : index;

    if (bitCount == 3)
    {
        constexpr float multipliers[4] =
        {
            0.8f,
            0.9f,
            1.0f,
            1.2f
        };

        return multipliers[magnitudeIndex];
    }

    throw std::invalid_argument("No Jayant multiplier table defined for this bit count.");
}
