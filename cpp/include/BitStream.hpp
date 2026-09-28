#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

class BitWriter
{
public:
    explicit BitWriter(uchar* data) :
        m_Data_(data)
    {
        assert(data != nullptr);
    }

    void Write(uchar value, uint8_t bitCount)
    {
        assert(bitCount > 0 && bitCount <= 8);

        m_BitBuffer_ |= static_cast<uint64_t>(value) << m_BitsInBuffer_;
        m_BitsInBuffer_ += bitCount;

        while (m_BitsInBuffer_ >= 8)
        {
            m_Data_[m_ByteIndex_++] = static_cast<uchar>(m_BitBuffer_ & 0xFF);

            m_BitBuffer_ >>= 8;
            m_BitsInBuffer_ -= 8;
        }
    }

    void Flush()
    {
        if (m_BitsInBuffer_ > 0)
        {
            m_Data_[m_ByteIndex_++] = static_cast<uchar>(m_BitBuffer_ & 0xFF);

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
        assert(bitCount > 0 && bitCount <= 8);

        return (valueCount * bitCount + 7) / 8;
    }

private:
    uchar* m_Data_;

    uint64_t m_BitBuffer_ = 0;
    uint8_t m_BitsInBuffer_ = 0;
    size_t m_ByteIndex_ = 0;
};


class BitReader
{
public:
    explicit BitReader(const uchar* data) :
        m_Data_(data)
    {
        assert(data != nullptr);
    }

    uchar Read(uint8_t bitCount)
    {
        assert(bitCount > 0 && bitCount <= 8);

        while (m_BitsInBuffer_ < bitCount)
        {
            m_BitBuffer_ |= static_cast<uint64_t>(m_Data_[m_ByteIndex_++]) << m_BitsInBuffer_;

            m_BitsInBuffer_ += 8;
        }

        const uint64_t mask = (1ULL << bitCount) - 1ULL;

        const uchar value = static_cast<uchar>(m_BitBuffer_ & mask);

        m_BitBuffer_ >>= bitCount;
        m_BitsInBuffer_ -= bitCount;

        return value;
    }

    size_t GetByteCount() const
    {
        return m_ByteIndex_;
    }

private:
    const uchar* m_Data_;

    uint64_t m_BitBuffer_ = 0;
    uint8_t m_BitsInBuffer_ = 0;
    size_t m_ByteIndex_ = 0;
};