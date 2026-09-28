#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

class LaplacianDCTQuantizer
{
public:
    static constexpr uint8_t MAX_BIT_COUNT = 8;

    LaplacianDCTQuantizer() = delete;

    LaplacianDCTQuantizer(uint8_t bitCount, float mean, float stddev)
        : m_BitCount_(bitCount),
          m_Mean_(mean),
          m_StandardDeviation_(stddev),
          m_Step_(laplacianSteps_[bitCount - 1] * stddev),
          m_LevelCount_(1u << bitCount)
    {
        assert(bitCount >= 1 && bitCount <= 8);
        assert(std::isfinite(mean));
        assert(std::isfinite(stddev));
        assert(stddev >= 0.0f);
    }

    uint8_t Quantize(float value) const
    {
        if (m_Step_ == 0.0f)
        {
            return static_cast<uint8_t>(m_LevelCount_ / 2);
        }

        const float position = (value - m_Mean_) / m_Step_ + static_cast<float>(m_LevelCount_) / 2.0f;

        return static_cast<uint8_t>(
            std::clamp(static_cast<int>(std::floor(position)), 0, static_cast<int>(m_LevelCount_) - 1)
        );
    }

    float Dequantize(uint8_t index) const
    {
        assert(index < m_LevelCount_);

        if (m_Step_ == 0.0f)
        {
            return m_Mean_;
        }

        return m_Mean_ + (static_cast<float>(index) + 0.5f - static_cast<float>(m_LevelCount_) / 2.0f) * m_Step_;
    }

private:
    inline static constexpr float laplacianSteps_[8] = {
        1.4140f,
        1.0873f,
        0.8707f,
        0.7309f,
        0.6334f,
        0.5613f,
        0.5055f,
        0.4609f
    };

    uint8_t m_BitCount_;
    float m_Mean_;
    float m_StandardDeviation_;
    float m_Step_;
    uint32_t m_LevelCount_;
};