#include <Quantizers/BTCQuantizer.hpp>
#include <BitStream.hpp>
#include <Quantizers/UniformQuantizer.hpp>

namespace
{
    double CalculateA(double q, double mean, double stdDev, double M)
    {
        return mean - stdDev * std::sqrt(q / (M - q));
    }
    
    double CalculateB(double q, double mean, double stdDev, double M)
    {
        return mean + stdDev * std::sqrt((M - q) / q); 
    }

    double CalculateMSE(const std::vector<uchar>& sortedPixels, double A, double B, size_t q)
    {
        const size_t M = sortedPixels.size();
        const size_t splitIndex = M - q;

        double squaredError = 0.0;

        // Lowest M-q pixels are represented by A
        for (size_t i = 0; i < splitIndex; ++i)
        {
            const double error = static_cast<double>(sortedPixels[i]) - A;
            squaredError += error * error;
        }

        // Highest q pixels are represented by B
        for (size_t i = splitIndex; i < M; ++i)
        {
            const double error = static_cast<double>(sortedPixels[i]) - B;
            squaredError += error * error;
        }

        return squaredError / static_cast<double>(M);
    }

    void WriteBlockBitmap(const cv::Mat& bloc, BitWriter& writer, uchar splitValue, size_t equalNeeded)
    {
        for (int y = 0; y < bloc.rows; ++y)
        {
            const uchar* row = bloc.ptr<uchar>(y);

            for (int x = 0; x < bloc.cols; ++x)
            {
                const uchar pixel = row[x];

                uchar bit = 0;

                if (pixel > splitValue)
                {
                    bit = 1;
                }
                else if (pixel == splitValue && equalNeeded > 0)
                {
                    bit = 1;
                    --equalNeeded;
                }

                writer.Write(bit, 1);
            }
        }
    }
}

uchar *BTCQuantizer::QuantizeAndPack(const ImageGray &image, uint8_t resconstructLevelBitCount, uint8_t blocSize, float& bitPerPixel)
{
    // Go trough all the blocs in the image
    const uint32_t blockCountX = image.GetWidth() / blocSize;
    const uint32_t blockCountY = image.GetHeight() / blocSize;
    const uint32_t blockCount = blockCountX * blockCountY;
    const size_t pixelCount = image.GetWidth() * image.GetHeight();

    BTCHeader header;
    header.width = image.GetWidth();
    header.height = image.GetHeight();
    header.blocSize = blocSize;
    header.resconstructLevelBitCount = resconstructLevelBitCount;

    // Every block has resconstructLevelBitCount * 2 + blocSize * blocSize bits
    const uint32_t bitPerBloc = resconstructLevelBitCount * 2 + blocSize * blocSize;
    const uint32_t dataBitCount = bitPerBloc * blockCount;
    const size_t packedDataSize = (dataBitCount + 7) / 8;

    const size_t totalDataSize = packedDataSize + sizeof(BTCHeader);
    bitPerPixel = static_cast<float>(totalDataSize) / static_cast<float>(pixelCount) * 8.0f;
    uchar* data = new uchar[totalDataSize];
    std::memcpy(data, &header, sizeof(BTCHeader));

    uchar* packedDataPtr = data + sizeof(BTCHeader);

    BitWriter writer(packedDataPtr);
    UniformPixelQuantizer uniformQuantizer(resconstructLevelBitCount);

    for (size_t y = 0; y < image.GetHeight(); y += blocSize)
    {
        for (size_t x = 0; x < image.GetWidth(); x += blocSize)
        {
            cv::Mat bloc = image.m_OpenCVImage_(cv::Rect(x, y, blocSize, blocSize));
            double A = 0.0;
            double B = 0.0;
            uchar splitValue = 0;
            size_t equalNeeded = 0;
            
            MinimizeMSE(bloc, A, B, splitValue, equalNeeded);
            
            // We must clamp beforehand
            uchar A8 = static_cast<uchar>(std::clamp(std::round(A), 0.0, 255.0));
            uchar B8 = static_cast<uchar>(std::clamp(std::round(B), 0.0, 255.0));
            
            uchar quantizedA = uniformQuantizer.Quantize(A8);
            uchar quantizedB = uniformQuantizer.Quantize(B8);

            writer.Write(quantizedA, resconstructLevelBitCount);
            writer.Write(quantizedB, resconstructLevelBitCount);
            WriteBlockBitmap(bloc, writer, splitValue, equalNeeded);
        }
    }
    writer.Flush();

    return data;
}

ImageGray BTCQuantizer::Unpack(uchar* packedData)
{
    // Read header
    BTCHeader header;
    std::memcpy(&header, packedData, sizeof(BTCHeader));

    const uint32_t width = header.width;
    const uint32_t height = header.height;
    const uint8_t blocSize = header.blocSize;
    const uint8_t reconstructLevelBitCount = header.resconstructLevelBitCount;

    // Packed BTC data starts immediately after the header.
    uchar* packedDataPtr = packedData + sizeof(BTCHeader);

    BitReader reader(packedDataPtr);
    UniformPixelQuantizer uniformQuantizer(reconstructLevelBitCount);

    cv::Mat image(height, width, CV_8UC1);

    for (size_t y = 0; y < height; y += blocSize)
    {
        for (size_t x = 0; x < width; x += blocSize)
        {
            // First values of every block are A and B.
            const uchar quantizedA = reader.Read(reconstructLevelBitCount);
            const uchar quantizedB = reader.Read(reconstructLevelBitCount);

            const uchar A = uniformQuantizer.Dequantize(quantizedA);
            const uchar B = uniformQuantizer.Dequantize(quantizedB);

            // The we have the N * N single bits
            for (size_t yBloc = 0; yBloc < blocSize; ++yBloc)
            {
                uchar* row = image.ptr<uchar>(y + yBloc);

                for (size_t xBloc = 0; xBloc < blocSize; ++xBloc)
                {
                    const uchar bit = reader.Read(1);

                    row[x + xBloc] = bit == 0 ? A : B;
                }
            }
        }
    }

    return ImageGray(image);
}

// Return A and B as doubles they will be quantized late.
void BTCQuantizer::MinimizeMSE(cv::Mat bloc, double& outA, double& outB, uchar& outSplitValue, size_t& outEqualNeeded)
{
    cv::Scalar imageMean;
    cv::Scalar imageStdDev;
    cv::meanStdDev(bloc, imageMean, imageStdDev);
    
    const double mean = imageMean[0];
    const double stdDev = imageStdDev[0];

    // Sort all of the pixel to save time in the MSE calculation
    const size_t pixelCount = bloc.rows * bloc.cols;

    std::vector<uchar> sortedPixels(pixelCount);
    
    size_t index = 0;
    for (int y = 0; y < bloc.rows; ++y)
    {
        // Get full row
        const uchar* row = bloc.ptr<uchar>(y);
        // Extract single values
        for (int x = 0; x < bloc.cols; ++x)
        {
            sortedPixels[index++] = row[x];
        }
    }
    std::sort(sortedPixels.begin(), sortedPixels.end());
    
    const double M = static_cast<double>(pixelCount);
    
    double smallestMSE = std::numeric_limits<double>::max();
    size_t bestQ = 0;
    // To simplify sort the values of the bloc
    
    for (size_t q = 1; q < M; ++q)
    {
        double qDouble = static_cast<double>(q);
        double A = CalculateA(qDouble, mean, stdDev, M);
        double B = CalculateB(qDouble, mean, stdDev, M);
        
        double MSE = CalculateMSE(sortedPixels, A, B, q);
        if (MSE < smallestMSE)
        {
            outA = A;
            outB = B;
            bestQ = q;
            smallestMSE = MSE;
        }
    }

    // There could be an edge case where [1, 2 ,3 ,3 ,3]
    //                                             ^
    // Where the MSE error would be lowest. Altough we ensured to get the best values for a and b
    // We are not splitting the array at the right position since both the pixels should have the exact same value
    
    const size_t splitIndex = pixelCount - bestQ;
    outSplitValue = sortedPixels[splitIndex];

    // Count pixels strictly greater than the split value.
    size_t greaterCount = 0;

    for (size_t i = splitIndex; i < pixelCount; ++i)
    {
        if (sortedPixels[i] > outSplitValue)
        {
            ++greaterCount;
        }
    }

    // Remaining pixels equal to splitValue that must be assigned to B.
    outEqualNeeded = bestQ - greaterCount;
}
