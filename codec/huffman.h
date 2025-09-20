#pragma once

#include <format>
#include <string>
#include <string_view>

namespace huffman {

std::string Compress(std::string_view raw);
std::string Decompress(std::string_view compressed);

template <int K>
std::string CompressMulti(std::string_view raw);
template <int K>
std::string DecompressMulti(std::string_view compressed);

template <int K>
std::string CompressMultiAvx512(std::string_view raw);
template <int K>
std::string DecompressMultiAvx512(std::string_view compressed);

// Specific implementations,
template <int K>
std::string CompressMultiAvx512Gather(std::string_view raw);
template <int K>
std::string CompressMultiAvx512Permute(std::string_view raw);
template <int K>
std::string DecompressMultiAvx512Gather(std::string_view compressed);
template <int K>
std::string DecompressMultiAvx512Permute(std::string_view compressed);

class HuffmanCompressor {
 public:
  static std::string Compress(std::string_view raw) {
    return ::huffman::Compress(raw);
  }
  static std::string Decompress(std::string_view compressed) {
    return ::huffman::Decompress(compressed);
  }
  static std::string name() { return "Huffman"; }
};

template <int K>
class HuffmanCompressorMulti {
 public:
  static std::string Compress(std::string_view raw) {
    return CompressMulti<K>(raw);
  }
  static std::string Decompress(std::string_view compressed) {
    return DecompressMulti<K>(compressed);
  }
  static std::string name() { return std::format("HuffmanMulti<{}>", K); }
};

template <int K>
class HuffmanCompressorAvx {
 public:
  static_assert(K % 8 == 0,
                "K must be a multiple of 8 in HuffmanCompressorAvx<K>");
  static std::string Compress(std::string_view raw) {
    return CompressMultiAvx512<K>(raw);
  }
  static std::string Decompress(std::string_view compressed) {
    return DecompressMultiAvx512<K>(compressed);
  }

  static std::string name() { return std::format("HuffmanAvx<{}>", K); }
};

template <int K>
class HuffmanCompressorAvxGather {
 public:
  static_assert(K % 8 == 0,
                "K must be a multiple of 8 in HuffmanCompressorAvxGather<K>");
  static std::string Compress(std::string_view raw) {
    return CompressMultiAvx512Gather<K>(raw);
  }
  static std::string Decompress(std::string_view compressed) {
    return DecompressMultiAvx512Gather<K>(compressed);
  }

  static std::string name() { return std::format("HuffmanAvxGather<{}>", K); }
};

template <int K>
class HuffmanCompressorAvxPermute {
 public:
  static_assert(K % 8 == 0,
                "K must be a multiple of 8 in HuffmanCompressorAvxPermute<K>");
  static std::string Compress(std::string_view raw) {
    return CompressMultiAvx512Permute<K>(raw);
  }
  static std::string Decompress(std::string_view compressed) {
    return DecompressMultiAvx512Permute<K>(compressed);
  }

  static std::string name() { return std::format("HuffmanAvxPermute<{}>", K); }
};

}  // namespace huffman
