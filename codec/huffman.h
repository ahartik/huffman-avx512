#pragma once

#include <array>
#include <string>
#include <format>
#include <string_view>

namespace huffman {

std::string Compress(std::string_view raw);
std::string Decompress(std::string_view compressed);

template <int K>
std::string CompressMulti(std::string_view raw);
template <int K>
std::string DecompressMulti(std::string_view compressed);


std::string DecompressMulti8Avx512(std::string_view compressed);

class HuffmanCompressor {
 public:
  static std::string Compress(std::string_view raw) {
    return ::huffman::Compress(raw);
  }
  static std::string Decompress(std::string_view compressed) {
    return ::huffman::Decompress(compressed);
  }
  static std::string name() {
    return "Huffman";
  }
};

template<int K> 
class HuffmanCompressorMulti {
 public:
  static std::string Compress(std::string_view raw) {
    return CompressMulti<K>(raw);
  }
  static std::string Decompress(std::string_view compressed) {
    return DecompressMulti<K>(compressed);
  }
  static std::string name() {
    return std::format("HuffmanMulti<{}>", K);
  }
};

class HuffmanCompressorAvx {
 public:
  static std::string Compress(std::string_view raw) {
    return CompressMulti<8>(raw);
  }
  static std::string Decompress(std::string_view compressed) {
    return DecompressMulti8Avx512(compressed);
  }
  static std::string name() {
    return "HuffmanAvx";
  }
};

namespace internal {

void CountSymbols(std::string_view text, int* sym_count);

} // namespace internal;

}  // namespace huffman
