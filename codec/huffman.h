#pragma once

#include <array>
#include <string>
#include <string_view>

using ByteCounts = std::array<int, 256>;

namespace huffman {

std::string Compress(std::string_view raw);
std::string Decompress(std::string_view compressed);

template <int K>
std::string CompressMulti(std::string_view raw);
template <int K>
std::string DecompressMulti(std::string_view compressed);

class HuffmanCompressor {
 public:
  static std::string Compress(std::string_view raw);
  static std::string Decompress(std::string_view compressed);
};

template<int> 
class HuffmanCompressorMulti {
 public:
  static std::string Compress(std::string_view raw);
  static std::string Decompress(std::string_view compressed);
};

}  // namespace huffman
