// Various byte-counting methods for building the symbol frequency histogram.

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace huffman {
using ByteHistogram = std::array<uint32_t, 256>;

ByteHistogram MakeHistogram(std::string_view str);

// Various versions which are included for separate testing and benchmarks.
// MakeHistogram(str) tries to pick the fastest from these.
ByteHistogram MakeHistogramSimple(std::string_view str);
ByteHistogram MakeHistogramMulti(std::string_view str);
ByteHistogram MakeHistogramVectorized(std::string_view str);
ByteHistogram MakeHistogramGatherScatter(std::string_view str);

}  // namespace huffman
