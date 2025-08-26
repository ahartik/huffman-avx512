#pragma once

#include <array>
#include <format>
#include <string>
#include <string_view>

namespace huffman {

std::string Huff0Compress(std::string_view raw);
std::string Huff0Decompress(std::string_view compressed);

class Huff0Compressor {
 public:
  static std::string Compress(std::string_view raw) {
    return ::huffman::Huff0Compress(raw);
  }
  static std::string Decompress(std::string_view compressed) {
    return ::huffman::Huff0Decompress(compressed);
  }
  static std::string name() { return "Huff0"; }
};

}  // namespace huffman
