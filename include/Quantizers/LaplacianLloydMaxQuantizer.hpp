#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

class LaplacianLloydMaxQuantizer
{
public:
    static constexpr uint8_t MAX_BIT_COUNT = 8;

    LaplacianLloydMaxQuantizer(uint8_t bitCount, float mean, float stddev)
        : mean_(mean),
          stddev_(stddev),
          levels_(GetLevels(bitCount))
    {
    }

    uint8_t Quantize(float value) const
    {
        if (stddev_ == 0.0f)
        {
            return 0;
        }

        // Normalize the coefficient: z = (x - mean) / stddev.
        const double normalized = (static_cast<double>(value) - mean_) / stddev_;

        // Find the first reconstruction level greater than z.
        const std::vector<double>::const_iterator upper = std::lower_bound(levels_.begin(), levels_.end(), normalized);

        if (upper == levels_.begin())
        {
            return 0;
        }

        if (upper == levels_.end())
        {
            return static_cast<uint8_t>(levels_.size() - 1);
        }

        const size_t upperIndex = static_cast<size_t>(upper - levels_.begin());

        // Lloyd-Max decision boundary: t = (y_left + y_right) / 2.
        const double boundary = (levels_[upperIndex - 1] + levels_[upperIndex]) / 2.0;

        if (normalized < boundary)
        {
            return static_cast<uint8_t>(upperIndex - 1);
        }

        return static_cast<uint8_t>(upperIndex);
    }

    float Dequantize(uint8_t index) const
    {
        if (index >= levels_.size())
        {
            throw std::out_of_range("Invalid quantization index");
        }

        // Undo normalization: reconstructed x = mean + stddev * level.
        return static_cast<float>(mean_ + stddev_ * levels_[index]);
    }

private:
    // For a Laplace distribution, variance = 2*b^2.
    // With standard deviation 1: b = 1 / sqrt(2).
    static constexpr double SCALE = 0.7071067811865475244;

    static double IntervalMean(double lower, double upper)
    {
        const double width = upper - lower;

        // Mean of an exponential distribution restricted to [lower, upper]:
        //
        // E[X | lower <= X < upper]
        //   = lower + b - (upper - lower) / (e^((upper - lower)/b) - 1)
        //
        // expm1(v) calculates e^v - 1 accurately when v is small.
        return lower + SCALE - width / std::expm1(width / SCALE);
    }

    static std::vector<double> MakeLevels(uint8_t bitCount)
    {
        // The Laplace distribution is symmetric. Compute the positive
        // half of the levels, then mirror them to obtain the negative half.
        const size_t positiveCount = size_t{1} << (bitCount - 1);
        std::vector<double> positive(positiveCount);

        // Starting guesses from quantiles of the positive exponential half:
        //
        // F(x) = 1 - e^(-x/b)
        // F^(-1)(p) = -b * ln(1 - p)
        for (size_t i = 0; i < positiveCount; ++i)
        {
            const double probability = (static_cast<double>(i) + 0.5) / positiveCount;
            positive[i] = -SCALE * std::log1p(-probability);
        }

        // Repeat the two Lloyd-Max steps:
        // 1. Boundaries = midpoints between neighboring levels.
        // 2. Levels = means of their corresponding intervals.
        for (int iteration = 0; iteration < 4000; ++iteration)
        {
            std::vector<double> updated(positiveCount);
            double largestChange = 0.0;

            for (size_t i = 0; i < positiveCount; ++i)
            {
                // The positive half starts at zero.
                const double lower = i == 0 ? 0.0 : (positive[i - 1] + positive[i]) / 2.0;

                if (i + 1 == positiveCount)
                {
                    // The last interval has no upper boundary.
                    // Its conditional mean is E[X | X >= lower] = lower + b.
                    updated[i] = lower + SCALE;
                }
                else
                {
                    const double upper = (positive[i] + positive[i + 1]) / 2.0;

                    updated[i] = IntervalMean(lower, upper);
                }

                largestChange = std::max(largestChange, std::abs(updated[i] - positive[i]));
            }

            positive.swap(updated);

            if (largestChange < 1e-12)
            {
                break;
            }
        }

        std::vector<double> levels(positiveCount * 2);

        for (size_t i = 0; i < positiveCount; ++i)
        {
            levels[positiveCount - 1 - i] = -positive[i];
            levels[positiveCount + i] = positive[i];
        }

        return levels;
    }

    // Simple system to cache 
    static const std::vector<double>& GetLevels(uint8_t bitCount)
    {
        if (bitCount == 0 || bitCount > MAX_BIT_COUNT)
        {
            throw std::invalid_argument("Bit count must be between 1 and 8");
        }

        // Since this is static const, the code is only ran once. Every table are now cached for the duration of this program.
        static const std::array<std::vector<double>, 9> tables = [] {
            std::array<std::vector<double>, 9> result;

            for (uint8_t bits = 1; bits <= MAX_BIT_COUNT; ++bits)
            {
                result[bits] = MakeLevels(bits);
            }

            return result;
        }();

        return tables[bitCount];
    }

    float mean_;
    float stddev_;
    const std::vector<double>& levels_;
};