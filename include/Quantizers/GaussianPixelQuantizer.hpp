#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <cassert>

// Used for DPCM error quantizers. Since error can be in position and negative
class GaussianErrorQuantizer
{
public:
    GaussianErrorQuantizer() = delete;

    GaussianErrorQuantizer(uint8_t bitCount, float mean, float stdDev)
        : m_BitCount_(bitCount),
          m_Mean_(mean),
          m_StandardDeviation_(stdDev),
          m_Step_(gaussianSteps_[bitCount - 1] * stdDev),
          m_LevelCount_(1 << bitCount)
    {
        assert(bitCount >= 1 && bitCount <= 5);
        assert(std::isfinite(mean));
        assert(std::isfinite(stdDev));
        assert(stdDev >= 0.0f);
    }

    uint8_t Quantize(int error) const
    {
        if (m_Step_ == 0.0f)
        {
            return static_cast<uint8_t>(m_LevelCount_ / 2);
        }
        const float position = (static_cast<float>(error) - m_Mean_) / m_Step_ + static_cast<float>(m_LevelCount_) / 2.0f;
        const int index = std::clamp(static_cast<int>(std::floor(position)), 0, m_LevelCount_ - 1);
        return static_cast<uint8_t>(index);
    }

    int Dequantize(uint8_t index) const
    {
        assert(index < m_LevelCount_);

        if (m_Step_ == 0.0f)
        {
            return static_cast<int>(std::lround(m_Mean_));
        }

        const float reconstructed = m_Mean_ + (static_cast<float>(index) + 0.5f - static_cast<float>(m_LevelCount_) / 2.0f) * m_Step_;

        // DPCM prediction errors for an 8-bit image cannot
        // meaningfully exceed this range.
        return std::clamp(static_cast<int>(std::lround(reconstructed)), -255, 255);
    }

private:
    inline static constexpr float gaussianSteps_[5] = {1.596f, 0.9957f, 0.5860f, 0.3352f, 0.1881f};

    uint8_t m_BitCount_;
    float m_Mean_;
    float m_StandardDeviation_;
    float m_Step_;
    int m_LevelCount_;
};

class GaussianPixelQuantizer
{
public:
    GaussianPixelQuantizer() = delete;
    GaussianPixelQuantizer(uint8_t bitCount, float mean, float stddev)
        : m_BitCount_(bitCount),
          m_Mean_(mean),
          m_StandardDeviation_(stddev),
          m_Step_(gaussianSteps_[bitCount - 1] * stddev), // The table offered in the manual takes in consideration an stddev of 1 multiply the step according to the new stddev.
          m_LevelCount(1 << bitCount)
    {
        assert(bitCount >= 1 && bitCount <= 5);
        assert(std::isfinite(mean) && std::isfinite(stddev));
        assert(stddev >= 0.0f);
    }

    uint8_t Quantize(uint8_t pixel) const
    {
        if (m_Step_ == 0.0f)
        {
            return static_cast<uint8_t>(m_LevelCount / 2);
        }

        const float position = (static_cast<float>(pixel) - m_Mean_) / m_Step_ + static_cast<float>(m_LevelCount) / 2.0f;

        return static_cast<uint8_t>(std::clamp(static_cast<int>(std::floor(position)), 0, m_LevelCount - 1));
    }

    uint8_t Dequantize(uint8_t index) const
    {
        assert(index < m_LevelCount);

        if (m_Step_ == 0.0f)
        {
            return ToPixel(m_Mean_);
        }

        const float reconstructed = m_Mean_ + (static_cast<float>(index) + 0.5f - static_cast<float>(m_LevelCount) / 2.0f) * m_Step_;

        return ToPixel(reconstructed);
    }

private:
    static uint8_t ToPixel(float value)
    {
        return static_cast<uint8_t>(
            std::lround(std::clamp(value, 0.0f, 255.0f))
        );
    }

    inline static constexpr float gaussianSteps_[5] = {
        1.596f, 0.9957f, 0.5860f, 0.3352f, 0.1881f
    };

    uint8_t m_BitCount_;
    float m_Mean_;
    float m_StandardDeviation_;
    float m_Step_;
    int m_LevelCount;
};