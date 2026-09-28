#include <Quantizers/DCTQuantizer.hpp>
#include <Quantizers/LaplaceQuantizer.hpp>
#include <BitStream.hpp>
#include <optional>

// Get the bit allocation matrix
std::vector<uint8_t> CalculateBitAllocation(const std::vector<float>& variances, size_t coefficientCount, uint8_t bitsPerPixel)
{
    assert(variances.size() == coefficientCount);
    assert(bitsPerPixel >= 0.0f);
    uint8_t maxBits =  LaplacianDCTQuantizer::MAX_BIT_COUNT;

    const size_t totalBitCount = static_cast<size_t>(bitsPerPixel) * coefficientCount;

    std::vector<uint8_t> bitCounts(coefficientCount, 0);

    for (size_t bit = 0; bit < totalBitCount; bit++)
    {
        size_t bestIndex = coefficientCount;
        double bestDistortion = -1.0;

        for (size_t i = 0; i < coefficientCount; i++)
        {
            if (bitCounts[i] >= maxBits)
            {
                continue;
            }

            // https://web.stanford.edu/class/ee398a/handouts/papers/Goyal%20-%20Transform%20Coding.pdf
            // See the bit allocation section (Equation 9)
            const double distortion = static_cast<double>(variances[i]) * std::pow(2.0, -2.0 * bitCounts[i]);

            if (distortion > bestDistortion)
            {
                bestDistortion = distortion;
                bestIndex = i;
            }
        }

        if (bestIndex == coefficientCount)
        {
            break;
        }

        bitCounts[bestIndex]++;
    }

    return bitCounts;
}

// Stream to calculate the variance and means of the bloc indices
void StreamBloc(const cv::Mat& bloc, std::vector<double>& sums, std::vector<double>& squaredSums, size_t blocSize)
{
    assert(bloc.rows == static_cast<int>(blocSize));
    assert(bloc.cols == static_cast<int>(blocSize));
    assert(bloc.type() == CV_32FC1);

    for (size_t y = 0; y < blocSize; y++)
    {
        for (size_t x = 0; x < blocSize; x++)
        {
            const size_t index = y * blocSize + x;
            const double value = static_cast<double>(bloc.at<float>(y, x));

            sums[index] += value;
            squaredSums[index] += value * value;
        }
    }
}

PackedData DCTQuantizer::QuantizeAndPack(const ImageGray &image, uint8_t bitCount, uint8_t blocSize)
{
    // Go trough all the blocs in the image
    const uint32_t blockCountX = image.GetWidth() / blocSize;
    const uint32_t blockCountY = image.GetHeight() / blocSize;
    const uint32_t blockCount = blockCountX * blockCountY;
    const uint32_t coefficientCount =  blocSize * blocSize;
    const size_t pixelCount = image.GetWidth() * image.GetHeight();
    
    cv::Mat dctImage(image.GetHeight(), image.GetWidth(), CV_32FC1);
    std::vector<double> sums(blockCount, 0.0);
    std::vector<double> squaredSums(blockCount, 0.0);
    std::vector<cv::Mat> DCTValues(blockCount);

    // Obtain the DFT for all possible blocs
    uint blocID = 0;
    for (size_t y = 0; y < image.GetHeight(); y += blocSize)
    {
        for (size_t x = 0; x < image.GetWidth(); x += blocSize)
        {
            cv::Mat block = image.m_OpenCVImage_(cv::Rect(x, y, blocSize, blocSize));
            cv::Mat blockFloat;
            block.convertTo(blockFloat, CV_32F);
            cv::Mat dctBlock;
            cv::dct(blockFloat, dctBlock);
            DCTValues[blocID] = dctBlock;
            StreamBloc(dctBlock, sums, squaredSums, blocSize);
            blocID++;
        }
    }

    std::vector<float> means(coefficientCount);
    std::vector<float> variances(coefficientCount);
    // Calculate the mean and variance for every bloc indices
    for (size_t i = 0; i < coefficientCount; i++)
    {
        const double mean = sums[i] / static_cast<double>(blockCount);
        double variance = squaredSums[i] / static_cast<double>(blockCount) - mean * mean;
        // Ensure we dont obtains negative values beacause of floating point error
        variance = std::max(0.0, variance);
        means[i] = static_cast<float>(mean);
        variances[i] = static_cast<float>(variance);
    }

    std::vector<uint8_t> bitsPerPixel =  CalculateBitAllocation(variances, coefficientCount, bitCount);

    
    std::vector<std::optional<LaplacianDCTQuantizer>> quantizers(coefficientCount);

    for (size_t i = 0; i < coefficientCount; ++i)
    {
        if (bitsPerPixel[i] == 0)
        {
            continue;
        }

        quantizers[i].emplace(bitsPerPixel[i], means[i], std::sqrt(variances[i]));
    }

    // Bit packing const
    const size_t meansSize = coefficientCount * sizeof(float);
    const size_t variancesSize = coefficientCount * sizeof(float);
    const size_t bitCountsSize = coefficientCount * sizeof(uint8_t);
    const size_t metadataSize = sizeof(DCTHeader) + meansSize + variancesSize + bitCountsSize;
    size_t bitsPerBlock = 0;

    for (uint8_t bits : bitsPerPixel)
    {
        bitsPerBlock += bits;
    }

    const size_t dataBitCount = static_cast<size_t>(blockCount) * bitsPerBlock;
    const size_t packedDataSize = (dataBitCount + 7) / 8;
    const size_t totalDataSize = metadataSize + packedDataSize;

    uchar* data = new uchar[totalDataSize]{};
    
    uchar* writePtr = data;
    DCTHeader header;
    header.width = image.GetWidth();
    header.height = image.GetHeight();
    header.blocSize = blocSize;

    // Header
    std::memcpy(writePtr, &header, sizeof(DCTHeader));
    writePtr += sizeof(DCTHeader);
    
    // Mean matrix
    std::memcpy(writePtr, means.data(), meansSize);
    writePtr += meansSize;

    // Variance Matrix
    std::memcpy(writePtr, variances.data(), variancesSize);
    writePtr += variancesSize;

    // Bit allocation matrix
    std::memcpy(writePtr, bitsPerPixel.data(), bitCountsSize);
    writePtr += bitCountsSize;

    BitWriter writer(writePtr);


    for (size_t blockIndex = 0; blockIndex < DCTValues.size(); ++blockIndex)
    {
        const cv::Mat& dctBlock = DCTValues[blockIndex];

        for (size_t y = 0; y < blocSize; ++y)
        {
            for (size_t x = 0; x < blocSize; ++x)
            {
                const size_t coefficientIndex = y * blocSize + x;
                const uint8_t coefficientBitCount = bitsPerPixel[coefficientIndex];

                if (coefficientBitCount == 0)
                {
                    continue;
                }

                const float value = dctBlock.at<float>(y, x);

                const uint8_t quantized = quantizers[coefficientIndex]->Quantize(value);

                writer.Write(quantized, coefficientBitCount);
            }
        }
    }

    writer.Flush();

    float bitPerPixel = static_cast<float>(totalDataSize) / static_cast<float>(pixelCount) * 8.0f;
    return PackedData {data, totalDataSize, bitPerPixel};
}

ImageGray DCTQuantizer::Unpack(uchar* packedData)
{
    assert(packedData != nullptr);

    const uchar* readPtr = packedData;

    // -----------------------------
    // Read header
    // -----------------------------
    DCTHeader header;
    std::memcpy(&header, readPtr, sizeof(DCTHeader));
    readPtr += sizeof(DCTHeader);

    const uint32_t width = header.width;
    const uint32_t height = header.height;
    const uint8_t blocSize = header.blocSize;

    const uint32_t coefficientCount = static_cast<uint32_t>(blocSize) * blocSize;

    const uint32_t blockCountX = width / blocSize;
    const uint32_t blockCountY = height / blocSize;

    // Mean
    std::vector<float> means(coefficientCount);
    const size_t meansSize = coefficientCount * sizeof(float);
    std::memcpy(means.data(), readPtr, meansSize);
    readPtr += meansSize;
    
    // Variance
    std::vector<float> variances(coefficientCount);
    const size_t variancesSize = coefficientCount * sizeof(float);
    std::memcpy(variances.data(), readPtr, variancesSize);
    readPtr += variancesSize;


    // Bit matrix
    std::vector<uint8_t> bitsPerPixel(coefficientCount);
    const size_t bitCountsSize = coefficientCount * sizeof(uint8_t);
    std::memcpy(bitsPerPixel.data(), readPtr, bitCountsSize);
    readPtr += bitCountsSize;


    // Recreate quantizers
    std::vector<std::optional<LaplacianDCTQuantizer>> quantizers(coefficientCount);

    for (size_t i = 0; i < coefficientCount; ++i)
    {
        if (bitsPerPixel[i] == 0)
        {
            continue;
        }

        quantizers[i].emplace(bitsPerPixel[i], means[i], std::sqrt(variances[i]));
    }


    BitReader reader(readPtr);

    cv::Mat reconstructedImage(height, width, CV_8UC1);

    for (size_t blockY = 0; blockY < blockCountY; ++blockY)
    {
        for (size_t blockX = 0; blockX < blockCountX; ++blockX)
        {
            cv::Mat dctBlock(blocSize, blocSize, CV_32FC1);

            // Reconstruct DCT coefficients.
            for (size_t y = 0; y < blocSize; ++y)
            {
                for (size_t x = 0; x < blocSize; ++x)
                {
                    const size_t coefficientIndex = y * blocSize + x;

                    const uint8_t coefficientBitCount = bitsPerPixel[coefficientIndex];

                    if (coefficientBitCount == 0)
                    {
                        // Coefficient is empty use the mean
                        dctBlock.at<float>(y, x) = means[coefficientIndex];
                        continue;
                    }

                    const uint8_t quantized = reader.Read(coefficientBitCount);

                    dctBlock.at<float>(y, x) = quantizers[coefficientIndex]->Dequantize(quantized);
                }
            }

            // DCT inverse
            cv::Mat reconstructedBlockFloat;
            cv::idct(dctBlock, reconstructedBlockFloat);

            // Convert float DCT result back to uint8.
            cv::Mat reconstructedBlock;
            reconstructedBlockFloat.convertTo(reconstructedBlock, CV_8UC1);
            
            // Copy data into final image
            const size_t imageX = blockX * blocSize;
            const size_t imageY = blockY * blocSize;

            reconstructedBlock.copyTo(reconstructedImage(cv::Rect(static_cast<int>(imageX), static_cast<int>(imageY), blocSize,blocSize)));
        }
    }

    return ImageGray(reconstructedImage);
}
