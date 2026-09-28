#pragma once
#include <cassert>
#include <opencv2/opencv.hpp>


class UniformPixelQuantizer
{
public:
    explicit UniformPixelQuantizer(uint8_t bitCount) : m_BitCount_(bitCount)
    {
        assert(m_BitCount_ > 0 && m_BitCount_ < 9);
    }

    // We skip the table entirely. There is not really a need for it in uniform
    uint8_t Quantize(uchar pixel) const
    {
        const uint32_t levelCount = 1u << m_BitCount_;
        return static_cast<uint8_t>((static_cast<uint32_t>(pixel) * levelCount) >> 8);
    }

    uchar Dequantize(uint8_t index) const
    {
        const uint32_t levelCount = 1u << m_BitCount_;
        assert(index < levelCount);
    
        // Return the midpoint of the index's interval in [0, 256).
        return static_cast<uchar>((static_cast<uint32_t>(index) * 256u + 128u) / levelCount);
    }

private:
    uint8_t m_BitCount_;
};