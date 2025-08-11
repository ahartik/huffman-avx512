#include "codec/huffman.h"

#include <cstdlib>
#include <cstdio>

#include <format>
#include <string>
#include "gtest/gtest.h"

using std::string;

TEST(HuffmanTest, Hello) {
  string raw = "Hello World";

  string compressed = huffman::Compress(raw);
  std::cout << std::flush;
  for (int i = 0; i < compressed.size(); ++i) {
    printf("%02x ", int(uint8_t(compressed[i])));
    if (i % 8 == 7) {
      printf("\n");
    }
  }
  printf("\n");
  std::cout << std::flush;
  string decompressed = huffman::Decompress(compressed);
  EXPECT_EQ(raw, decompressed);
}

TEST(HuffmanTest, LongRandom) {
  string raw;
  int len = 1000;
  for (int i = 0; i < len; ++i) {
    // Make a biased distribution
    uint8_t ch = (rand() & rand()) & 0xff;
    raw.push_back(ch);
  }
  string compressed = huffman::Compress(raw);
  string decompressed = huffman::Decompress(compressed);
  EXPECT_EQ(raw, decompressed);
}

