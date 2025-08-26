#include "codec/huff0.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <format>
#include <iostream>

// Just for experimentation
#define HUF_STATIC_LINKING_ONLY

#include "fse/huf.h"

namespace huffman {
std::string Huff0Compress(std::string_view raw) {
  size_t bound = HUF_compressBound(raw.size());
  std::string buf(4 + bound, 0);

  size_t compressed_size =
      HUF_compress(buf.data() + 4, bound, raw.data(), raw.size());

  int32_t size_word = raw.size();
  if (compressed_size == 0) {
    // Negative sizes mean uncompressed data
    size_word = -size_word;
    // Just copy the data here:
    memcpy(buf.data() + 4, raw.data(), raw.size());
    compressed_size = raw.size();
  }
  memcpy(buf.data(), &size_word, 4);

  buf.resize(4 + compressed_size);
  return buf;
}
std::string Huff0Decompress(std::string_view compressed) {
  int32_t raw_size;
  memcpy(&raw_size, compressed.data(), 4);
  compressed.remove_prefix(4);
  if (raw_size <= 0) {
    return std::string(compressed);
  }

  std::string raw(raw_size, 0);
#if 0
  // Note: This is OK for benchmarking, but does not pass all tests.
  size_t decompressed_size = HUF_decompress4X1(
      raw.data(), raw_size, compressed.data(), compressed.size());
#else
  size_t decompressed_size = HUF_decompress(
      raw.data(), raw_size, compressed.data(), compressed.size());
#endif
  if (decompressed_size != size_t(raw_size)) {
    std::cout << std::format(
                     "decompressed_size={} while raw_size={}\n"
                     "Error name: {}\n",
                     decompressed_size, raw_size,
                     HUF_getErrorName(decompressed_size))
              << std::flush;
    abort();
  }
  return raw;
}
}  // namespace huffman
