#include "codec/huffman.h"

#include <cstdio>

#include <format>
#include <string>
#include "gtest/gtest.h"

using std::string;

TEST(HuffmanTest, Hello) {
  string raw = "Hello World";

  string compressed = huffman::compress(raw);
  std::cout << std::flush;
  for (int i = 0; i < compressed.size(); ++i) {
    printf("%02x ", int(uint8_t(compressed[i])));
    if (i % 8 == 7) {
      printf("\n");
    }
  }
  printf("\n");
  std::cout << std::flush;
  string decompressed = huffman::decompress(compressed);
  EXPECT_EQ(raw, decompressed);
}
