#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

class BitWriter
{
public:
    explicit BitWriter(uint8_t* data) : m_Data_(data)
    {
        if (data == nullptr)
        {
            throw std::invalid_argument("BitWriter data cannot be nullptr.");
        }
    }

    void Write(uint32_t value, uint8_t bitCount)
    {
        if (bitCount == 0 || bitCount > 32)
        {
            throw std::invalid_argument("bitCount must be between 1 and 32.");
        }

        m_BitBuffer_ |= static_cast<uint64_t>(value) << m_BitsInBuffer_;
        m_BitsInBuffer_ += bitCount;

        while (m_BitsInBuffer_ >= 8)
        {
            m_Data_[m_ByteIndex_++] = static_cast<uint8_t>(m_BitBuffer_ & 0xFF);

            m_BitBuffer_ >>= 8;
            m_BitsInBuffer_ -= 8;
        }
    }

    void Flush()
    {
        if (m_BitsInBuffer_ > 0)
        {
            m_Data_[m_ByteIndex_++] = static_cast<uint8_t>(m_BitBuffer_ & 0xFF);

            m_BitBuffer_ = 0;
            m_BitsInBuffer_ = 0;
        }
    }

    size_t GetByteCount() const
    {
        return m_ByteIndex_;
    }

    static size_t GetPackedSize(size_t valueCount, uint8_t bitCount)
    {
        return (valueCount * bitCount + 7) / 8;
    }

private:
    uint8_t* m_Data_;

    uint64_t m_BitBuffer_ = 0;
    uint8_t m_BitsInBuffer_ = 0;
    size_t m_ByteIndex_ = 0;
};


class BitReader
{
public:
    explicit BitReader(const uint8_t* data) : m_Data_(data)
    {
        if (data == nullptr)
        {
            throw std::invalid_argument("BitReader data cannot be nullptr.");
        }
    }

    uint32_t Read(uint8_t bitCount)
    {
        if (bitCount == 0 || bitCount > 32)
        {
            throw std::invalid_argument("bitCount must be between 1 and 32.");
        }

        while (m_BitsInBuffer_ < bitCount)
        {
            m_BitBuffer_ |= static_cast<uint64_t>(m_Data_[m_ByteIndex_++]) << m_BitsInBuffer_;
            m_BitsInBuffer_ += 8;
        }

        const uint64_t mask = (1ULL << bitCount) - 1;
        const uint32_t value = static_cast<uint32_t>(m_BitBuffer_ & mask);

        m_BitBuffer_ >>= bitCount;
        m_BitsInBuffer_ -= bitCount;

        return value;
    }

    size_t GetByteCount() const
    {
        return m_ByteIndex_;
    }

private:
    const uint8_t* m_Data_;

    uint64_t m_BitBuffer_ = 0;
    uint8_t m_BitsInBuffer_ = 0;
    size_t m_ByteIndex_ = 0;
};