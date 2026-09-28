#include <Quantizers/VectorQuantizer.hpp>
#include <BitStream.hpp>

UcharDict::UcharDict(uint8_t vectorCount, uint8_t vectorSize) :
    vectorCount(vectorCount), vectorSize(vectorSize)
{
    values = new uchar[static_cast<size_t>(vectorCount) * vectorSize]{};
}

UcharDict::UcharDict(const VectorArray &vectorArray) : 
    vectorCount(vectorArray.vectorCount), vectorSize(vectorArray.vectorSize)
{
    values = new uchar[static_cast<size_t>(vectorCount) * vectorSize]{};
    constexpr double min = std::numeric_limits<uchar>::min();
    constexpr double max = std::numeric_limits<uchar>::max();
    
    for (size_t i = 0; i < GetDataSize(); i++)
    {
        values[i] = static_cast<uchar>(std::clamp(vectorArray.values[i], min, max));
    }
}

UcharDict::~UcharDict()
{
    delete[] values;
}

UcharDict::UcharDict(UcharDict&& other) noexcept :
    values(other.values),
    vectorCount(other.vectorCount),
    vectorSize(other.vectorSize)
{
    other.values = nullptr;
}

UcharDict& UcharDict::operator=(UcharDict&& other) noexcept
{
    if (this != &other)
    {
        delete[] values;

        values = other.values;
        vectorCount = other.vectorCount;
        vectorSize = other.vectorSize;
        other.values = nullptr;
    }
    return *this;
}

uchar* UcharDict::operator[](uint32_t vectorIndex)
{
    assert(vectorIndex < vectorCount);
    return values + vectorIndex * vectorSize;
}

const uchar* UcharDict::operator[](uint32_t vectorIndex) const
{
    assert(vectorIndex < vectorCount);
    return values + vectorIndex * vectorSize;
}

size_t UcharDict::GetDataSize() const
{
    return static_cast<size_t>(vectorCount) * vectorSize;
}

VectorArray::VectorArray(uint8_t vectorCount, uint8_t vectorSize) :
    vectorCount(vectorCount),
    vectorSize(vectorSize)
{
    uint32_t size = vectorCount * vectorSize;
    values = new double[size];
    std::fill(values, values + size, 0.0);
}

VectorArray::~VectorArray()
{
    delete[] values;
}

VectorArray::VectorArray(VectorArray&& other) noexcept
    : vectorCount(other.vectorCount),
      vectorSize(other.vectorSize),
      values(other.values) 
{
    other.values = nullptr;
}

VectorArray& VectorArray::operator=(VectorArray&& other) noexcept
{
    if (this != &other)
    {
        delete[] values;

        values = other.values;
        vectorCount = other.vectorCount;
        vectorSize = other.vectorSize;

        other.values = nullptr;
    }

    return *this;
}

double* VectorArray::operator[](uint32_t vectorIndex)
{
    assert(vectorIndex < vectorCount);
    return values + vectorIndex * vectorSize;
}

const double* VectorArray::operator[](uint32_t vectorIndex) const
{
    assert(vectorIndex < vectorCount);
    return values + vectorIndex * vectorSize;
}

VectorArray VectorArray::DoubleVectors()
{
    constexpr int PERTURBATION = 1;
    uint32_t newVectorCount = vectorCount * 2;
    VectorArray newVectorArray(newVectorCount, vectorSize);

    for (size_t i = 0; i < vectorCount; i++)
    {
        double* currentVector = (*this)[i];
        double* newVector1 = newVectorArray[i * 2];
        double* newVector2 = newVectorArray[i * 2 + 1];

        for (size_t j = 0; j < vectorSize; j++)
        {
            newVector1[j] = std::min(255.0, currentVector[j] + PERTURBATION);

            newVector2[j] = std::max(0.0, currentVector[j] - PERTURBATION);
        }
    }
    return newVectorArray;
}

uchar* VectorQuantizer::QuantizeAndPack(const ImageGray& image, uint8_t vectorBitCount, uint8_t blocSize)
{
    // Train dictionary
    VectorArray vectorArray = TrainLBG(image, vectorBitCount, blocSize);

    // Find dictionary index for every image block
    uint32_t blockCount = 0;

    uchar* blockIndices = FindBlockIndices(image, vectorArray, blocSize, blockCount);

    // Convert floating-point centroids to uchar
    UcharDict ucharDict(vectorArray);

    // Number of bytes required for packed indices
    const size_t indexBitCount = static_cast<size_t>(blockCount) * vectorBitCount;
    const size_t packedIndexSize = (indexBitCount + 7) / 8;
    const size_t fileSize = sizeof(VectorQuantizedImageHeader) + ucharDict.GetDataSize() + packedIndexSize;

    // Zero initialize because the last byte may only be partially used
    uchar* fullFile = new uchar[fileSize]{};

    // Header
    VectorQuantizedImageHeader header;
    header.width = image.GetWidth();
    header.height = image.GetHeight();
    header.vectorBitCount = vectorBitCount;
    header.blocSize = blocSize;
    std::memcpy(fullFile, &header, sizeof(VectorQuantizedImageHeader));

    // Dictionary
    uchar* dictionaryData = fullFile + sizeof(VectorQuantizedImageHeader);
    std::memcpy(dictionaryData, ucharDict.values, ucharDict.GetDataSize());
    
    // Packed block indices
    uchar* packedIndices = dictionaryData + ucharDict.GetDataSize();

    BitWriter writer(packedIndices);

    for (uint32_t i = 0; i < blockCount; ++i)
    {
        writer.Write(blockIndices[i], vectorBitCount);
    }
    writer.Flush();

    delete[] blockIndices;

    return fullFile;
}

ImageGray VectorQuantizer::Unpack(uchar* packedData)
{
    if (packedData == nullptr)
    {
        throw std::invalid_argument("packedData cannot be nullptr.");
    }

    // Header
    VectorQuantizedImageHeader header;

    std::memcpy(&header, packedData, sizeof(VectorQuantizedImageHeader));
    const uint32_t width = header.width;
    const uint32_t height = header.height;
    const uint8_t vectorBitCount = header.vectorBitCount;
    const uint8_t blocSize = header.blocSize;

    // Dictionary information

    const uint32_t vectorCount = 1u << vectorBitCount;

    const uint32_t vectorSize = static_cast<uint32_t>(blocSize) * blocSize;
    const size_t dictionarySize = static_cast<size_t>(vectorCount) * vectorSize;
    const uchar* dictionaryData = packedData + sizeof(VectorQuantizedImageHeader);

    // Packed indices start immediately after dictionary
    const uchar* packedIndices = dictionaryData + dictionarySize;

    // Output image
    cv::Mat reconstructedImage(height, width, CV_8UC1 );

    const uint32_t blockCountX = width / blocSize;
    const uint32_t blockCountY = height / blocSize;
    const uint32_t blockCount = blockCountX * blockCountY;

    // ------------------------------------------------------------
    // Unpack indices and reconstruct blocks
    // ------------------------------------------------------------

    const uint64_t indexMask = (1ULL << vectorBitCount) - 1ULL;
    BitReader reader(packedIndices);

    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        uchar value = reader.Read(vectorBitCount);

        // Dictionary vector
        const uchar* vector = dictionaryData + static_cast<size_t>(value) * vectorSize;

        // Reconstruct block
        uint32_t pixelIndex = 0;

        // Find where this block belongs in the image
        const uint32_t blockX = (blockIndex % blockCountX) * blocSize;
        const uint32_t blockY = (blockIndex / blockCountX) * blocSize;
        
        for (uint32_t yBlock = 0; yBlock < blocSize; ++yBlock)
        {
            uchar* row = reconstructedImage.ptr<uchar>(blockY + yBlock) + blockX;

            for (uint32_t xBlock = 0; xBlock < blocSize; ++xBlock)
            {
                row[xBlock] = vector[pixelIndex++];
            }
        }
    }

    return ImageGray(reconstructedImage);
}


// Linde–Buzo–Gray
VectorArray VectorQuantizer::TrainLBG(const ImageGray& image, uint8_t vectorBitCount, uint8_t blocSize)
{
    if (image.GetWidth() % blocSize != 0 || image.GetHeight() % blocSize != 0)
    {
        throw std::invalid_argument("Image dimensions must be divisible by blocSize.");
    }

    const uint32_t totalWantedVector = 1u << vectorBitCount;
    const uint32_t vectorSize = blocSize * blocSize;

    // LBG starts with ONE centroid.
    VectorArray vectorArray(1, vectorSize);

    // With one vector, this calculates the global centroid.
    double previousDistortion = RecalculateCentroids(image, vectorArray, blocSize);

    while (vectorArray.vectorCount < totalWantedVector)
    {
        vectorArray = vectorArray.DoubleVectors();

        previousDistortion = RecalculateCentroids(image, vectorArray, blocSize);

        while (true)
        {
            if (previousDistortion == 0.0)
            {
                break;
            }

            double distortion = RecalculateCentroids(image, vectorArray, blocSize);

            double improvement = (previousDistortion - distortion) / previousDistortion;

            if (improvement < 0.001)
            {
                break;
            }

            previousDistortion = distortion;
        }
    }

    return vectorArray;
}

double VectorQuantizer::RecalculateCentroids(const ImageGray& image, VectorArray& vectorArray, uint8_t blocSize)
{
    const uint32_t vectorSize = vectorArray.vectorSize;
    const uint32_t vectorCount = vectorArray.vectorCount;

    // New recalculated centroids
    VectorArray centroidSums(vectorCount, vectorSize);
    
    // Temporary accumulators for each centroid.
    uint32_t* centroidCounts = new uint32_t[vectorCount]{};
    double totalDistortion = 0.0;
    for (size_t y = 0; y < image.GetHeight(); y += blocSize)
    {
        for (size_t x = 0; x < image.GetWidth(); x += blocSize)
        {
            cv::Mat block = image.m_OpenCVImage_(cv::Rect(x, y, blocSize, blocSize));

            uint32_t closestVector = 0;
            double closestDistance = std::numeric_limits<double>::max();

            for (uint32_t vectorIndex = 0; vectorIndex < vectorCount; ++vectorIndex)
            {
                const double* centroid = vectorArray[vectorIndex];

                double distance = 0.0;
                uint32_t pixelIndex = 0;

                for (uint32_t yBlock = 0; yBlock < blocSize; ++yBlock)
                {
                    for (uint32_t xBlock = 0; xBlock < blocSize; ++xBlock)
                    {
                        double pixelValue = block.at<uchar>(yBlock, xBlock);

                        const double difference = pixelValue - centroid[pixelIndex];
                        // Skip the sqrt, it is not needed at all
                        distance += difference * difference;
                        ++pixelIndex;
                    }
                }
                
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestVector = vectorIndex;
                }
            }

            // Add block to the winning centroid
            double* nearestVector = centroidSums[closestVector];
            totalDistortion += closestDistance;
            uint32_t pixelIndex = 0;

            for (uint32_t yBlock = 0; yBlock < blocSize; ++yBlock)
            {
                for (uint32_t xBlock = 0; xBlock < blocSize; ++xBlock)
                {
                    nearestVector[pixelIndex] += block.at<uchar>(yBlock, xBlock);
                    ++pixelIndex;
                }
            }
            ++centroidCounts[closestVector];
        }

    }

    // Calculate the new centroid values.
    for (uint32_t vectorIndex = 0; vectorIndex < vectorCount; ++vectorIndex)
    {
        if (centroidCounts[vectorIndex] == 0)
        {
            // TODO Ideally we would want to do something with emnpty clusters
            // Since they are basicly just dead values
            continue;
        }
        
        double* centroidSum = centroidSums[vectorIndex];
        const double inverseCount = 1.0 / static_cast<double>(centroidCounts[vectorIndex]);
        double* originalVector = vectorArray[vectorIndex]; 
        
        for (uint32_t i = 0; i < vectorSize; ++i)
        {
            originalVector[i] = centroidSum[i] * inverseCount;
        }
    }
    
    delete[] centroidCounts;
    return totalDistortion;
}


uchar* VectorQuantizer::FindBlockIndices(const ImageGray& image, const VectorArray& vectorArray, uint8_t blocSize, uint32_t& blockCount)
{
    const uint32_t blockCountX = image.GetWidth() / blocSize;
    const uint32_t blockCountY = image.GetHeight() / blocSize;
    blockCount = blockCountX * blockCountY;

    uchar* blockIndices = new uchar[blockCount];

    uint32_t blockIndex = 0;

    for (size_t y = 0; y < image.GetHeight(); y += blocSize)
    {
        for (size_t x = 0; x < image.GetWidth(); x += blocSize)
        {
            cv::Mat block = image.m_OpenCVImage_(cv::Rect(x, y, blocSize, blocSize));

            uint32_t closestVector = 0;
            double closestDistance = std::numeric_limits<double>::max();

            for (uint32_t vectorIndex = 0; vectorIndex < vectorArray.vectorCount; ++vectorIndex)
            {
                const double* centroid = vectorArray[vectorIndex];

                double distance = 0.0;
                uint32_t pixelIndex = 0;

                for (uint32_t yBlock = 0; yBlock < blocSize; ++yBlock)
                {
                    for (uint32_t xBlock = 0; xBlock < blocSize; ++xBlock)
                    {
                        const double pixelValue = block.at<uchar>(yBlock, xBlock);
                        const double difference = pixelValue - centroid[pixelIndex];

                        distance += difference * difference;
                        ++pixelIndex;
                    }
                }

                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestVector = vectorIndex;
                }
            }

            blockIndices[blockIndex] = closestVector;
            ++blockIndex;
        }
    }

    return blockIndices;
}