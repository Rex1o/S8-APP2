#include <Quantizers/ScalarQuantizer.hpp>
#include <BitStream.hpp>

// We should try with a Floyd–Steinberg dithering seems like we could have a way better image
uchar *UniformQuantizer::QuantizeAndPack(const ImageGray& image, uint8_t bitCount)
{
    if (bitCount == 0 || bitCount > 8)
    {
        throw std::invalid_argument("M must be between 1 and 8.");
    }

    const uint32_t levels = 1u << bitCount;
    const size_t dataBits = image.GetHeight() *  image.GetWidth() * bitCount;
    const size_t packedDataSize = (dataBits + 7) / 8;

    size_t outputSize = sizeof(ScalarQuantizedImageHeader) + packedDataSize;
    unsigned char* output = new unsigned char[outputSize];

    ScalarQuantizedImageHeader header;
    header.width = image.GetWidth();
    header.height = image.GetHeight();
    header.bitsPerPixel = bitCount;

    std::memcpy(output, &header, sizeof(header));

    // Packed image starts after header.
    unsigned char* packedData = output + sizeof(header);

    BitWriter writer(packedData);

    for (uint y = 0; y < image.GetHeight(); ++y)
    {
        for (uint x = 0; x < image.GetWidth(); ++x)
        {
            uchar pixel = image.m_OpenCVImage_.at<uchar>(y, x);
            uint32_t quantized = (static_cast<uint32_t>(pixel) * levels) >> 8;

            writer.Write(quantized, bitCount);
        }
    }
    writer.Flush();

    return output;
}

ImageGray UniformQuantizer::Unpack(uchar* packedData)
{
    ScalarQuantizedImageHeader header;
    std::memcpy(&header, packedData, sizeof(header));

    uint32_t width = header.width;
    uint32_t height = header.height;
    uchar bitCount = header.bitsPerPixel;
    uchar* pixels = packedData + sizeof(ScalarQuantizedImageHeader);

    if (bitCount == 0 || bitCount > 8)
    {
        throw std::invalid_argument("Invalid bits per pixel.");
    }

    const size_t pixelCount = static_cast<size_t>(width) * height;
    const size_t dataBits = pixelCount * bitCount;
    
    cv::Mat newImage(height, width, CV_8UC1);
    const uint32_t mask = (1u << bitCount) - 1u;
    
    BitReader reader(pixels);
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            uint32_t quantized = reader.Read(header.bitsPerPixel);
            newImage.at<uchar>(y,x) = static_cast<uchar>((quantized * 255u) / mask); 
        }
    }

    return ImageGray(newImage);
}

namespace
{
    // Approximation of inverse standard-normal CDF.
    // Peter J. Acklam's rational approximation.
    // TODO revenir ici et comprendre ceci
    double InverseNormalCDF(double p)
    {
        if (p <= 0.0 || p >= 1.0)
        {
            throw std::invalid_argument("p must be in (0, 1)");
        }

        constexpr double a1 = -3.969683028665376e+01;
        constexpr double a2 =  2.209460984245205e+02;
        constexpr double a3 = -2.759285104469687e+02;
        constexpr double a4 =  1.383577518672690e+02;
        constexpr double a5 = -3.066479806614716e+01;
        constexpr double a6 =  2.506628277459239e+00;

        constexpr double b1 = -5.447609879822406e+01;
        constexpr double b2 =  1.615858368580409e+02;
        constexpr double b3 = -1.556989798598866e+02;
        constexpr double b4 =  6.680131188771972e+01;
        constexpr double b5 = -1.328068155288572e+01;

        constexpr double c1 = -7.784894002430293e-03;
        constexpr double c2 = -3.223964580411365e-01;
        constexpr double c3 = -2.400758277161838e+00;
        constexpr double c4 = -2.549732539343734e+00;
        constexpr double c5 =  4.374664141464968e+00;
        constexpr double c6 =  2.938163982698783e+00;

        constexpr double d1 =  7.784695709041462e-03;
        constexpr double d2 =  3.224671290700398e-01;
        constexpr double d3 =  2.445134137142996e+00;
        constexpr double d4 =  3.754408661907416e+00;

        constexpr double pLow = 0.02425;
        constexpr double pHigh = 1.0 - pLow;

        if (p < pLow)
        {
            double q = std::sqrt(-2.0 * std::log(p));

            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }

        if (p <= pHigh)
        {
            double q = p - 0.5;
            double r = q * q;

            return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        }

        double q = std::sqrt(-2.0 * std::log(1.0 - p));

        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    std::vector<float> CreateGaussianLevels(float mean, float variance, uint32_t levelCount)
    {
        std::vector<float> levels(levelCount);

        if (variance <= 0.0f)
        {
            std::fill(levels.begin(), levels.end(), mean);
            return levels;
        }

        float standardDeviation = std::sqrt(variance);

        for (uint32_t i = 0; i < levelCount; ++i)
        {
            // Center of each equal-probability region.
            double probability = (static_cast<double>(i) + 0.5) / static_cast<double>(levelCount);

            levels[i] = mean + standardDeviation * static_cast<float>(InverseNormalCDF(probability));
            levels[i] = std::clamp(levels[i], 0.0f, 255.0f);
        }

        return levels;
    }
}

uchar* FowardAdaptiveGaussQuantizer::QuantizeAndPack(const ImageGray& image, uint8_t bitCount)
{
    if (bitCount == 0 || bitCount > 8)
    {
        throw std::invalid_argument("bitCount must be between 1 and 8.");
    }

    const uint32_t levels = 1u << bitCount;
    const size_t pixelCount = image.GetWidth() * image.GetHeight();
    const size_t dataBits = pixelCount * bitCount;
    const size_t packedDataSize = (dataBits + 7) / 8;

    float mean;
    float variance;

    image.GetMeanAndVariance(mean, variance);

    AdaptiveGaussQuantizedImageHeader header;
    header.width = image.GetWidth();
    header.height = image.GetHeight();
    header.mean = mean;
    header.variance = variance;
    header.bitsPerPixel = bitCount;

    const size_t outputSize = sizeof(AdaptiveGaussQuantizedImageHeader) + packedDataSize;

    uchar* output = new uchar[outputSize];

    std::memcpy(output, &header, sizeof(header));

    uchar* packedData = output + sizeof(header);
    std::memset(packedData, 0, packedDataSize);

    std::vector<float> reconstructionLevels = CreateGaussianLevels(mean, variance, levels);

    // Decision thresholds halfway between reconstruction levels.
    std::vector<float> thresholds(levels - 1);

    for (uint32_t i = 0; i < levels - 1; ++i)
    {
        thresholds[i] = (reconstructionLevels[i] + reconstructionLevels[i + 1]) * 0.5f;
    }

    uint64_t bitBuffer = 0;
    uint8_t bitsInBuffer = 0;
    size_t outputIndex = 0;

    for (uint32_t y = 0; y < image.GetHeight(); ++y)
    {
        for (uint32_t x = 0; x < image.GetWidth(); ++x)
        {
            uchar pixel = image.m_OpenCVImage_.at<uchar>(y, x);

            uint32_t quantized = 0;
            
            // TODO a binary serach here could be much faster.
            while (quantized < thresholds.size() && pixel >= thresholds[quantized])
            {
                ++quantized;
            }

            bitBuffer |= static_cast<uint64_t>(quantized) << bitsInBuffer;
            bitsInBuffer += bitCount;

            while (bitsInBuffer >= 8)
            {
                packedData[outputIndex++] = static_cast<uchar>(bitBuffer & 0xFF);

                bitBuffer >>= 8;
                bitsInBuffer -= 8;
            }
        }
    }

    if (bitsInBuffer > 0)
    {
        packedData[outputIndex] =
            static_cast<uchar>(bitBuffer & 0xFF);
    }

    return output;
}

ImageGray FowardAdaptiveGaussQuantizer::Unpack(uchar* packedData)
{
    AdaptiveGaussQuantizedImageHeader header;

    std::memcpy(&header, packedData, sizeof(header));

    if (header.bitsPerPixel == 0 || header.bitsPerPixel > 8)
    {
        throw std::runtime_error("Invalid bitsPerPixel.");
    }

    const uint32_t levels = 1u << header.bitsPerPixel;

    std::vector<float> reconstructionLevels = CreateGaussianLevels(header.mean, header.variance, levels);

    cv::Mat image(header.height, header.width, CV_8UC1);

    uchar* compressedData = packedData + sizeof(AdaptiveGaussQuantizedImageHeader);

    uint64_t bitBuffer = 0;
    uint8_t bitsInBuffer = 0;
    size_t inputIndex = 0;

    const uint64_t mask = (1ULL << header.bitsPerPixel) - 1ULL;

    for (uint32_t y = 0; y < header.height; ++y)
    {
        for (uint32_t x = 0; x < header.width; ++x)
        {
            if (bitsInBuffer < header.bitsPerPixel)
            {
                bitBuffer |= static_cast<uint64_t>(compressedData[inputIndex++]) << bitsInBuffer;
                bitsInBuffer += 8;
            }

            uint32_t quantized = static_cast<uint32_t>(bitBuffer & mask);

            bitBuffer >>= header.bitsPerPixel;
            bitsInBuffer -= header.bitsPerPixel;

            float reconstructed = reconstructionLevels[quantized];

            image.at<uchar>(y, x) = cv::saturate_cast<uchar>(std::round(reconstructed));
        }
    }

    return ImageGray(image);
}

uchar *JayantQuantizer::QuantizeAndPack(const ImageGray &image, uint8_t bitCount)
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

            // IMPORTANT:
            // q[n] changes delta for pixel n+1.
            delta *= GetMultiplier(quantized, bitCount);

            delta = std::clamp(delta, MIN_DELTA, MAX_DELTA);
        }
    }

    if (bitsInBuffer > 0)
    {
        outputData[outputIndex] = static_cast<uchar>(bitBuffer & 0xFF);
    }

    return output;
}

ImageGray JayantQuantizer::Unpack(uchar *packedData)
{
    JayantQuantizedImageHeader header;
    // TODO we could simply reinterpret cast
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
