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
  int len = 2000;
  for (int i = 0; i < len; ++i) {
    uint8_t ch = 0;
    do {
      // Make a biased distribution
      ch = (rand() & rand() & rand()) & 0xff;
    } while(!std::isprint(ch));
    raw.push_back(ch);
  }
  string compressed = huffman::Compress(raw);
  string decompressed = huffman::Decompress(compressed);
  EXPECT_EQ(raw, decompressed);
}

TEST(HuffmanTest, SingleSymbolOnly) {
  string compressed = huffman::Compress("AAA");
  string decompressed = huffman::Decompress(compressed);
  EXPECT_EQ("AAA", decompressed);
}

#if 1
TEST(HuffmanTest, EmptyString) {
  string compressed = huffman::Compress("");
  string decompressed = huffman::Decompress(compressed);
  EXPECT_EQ("", decompressed);
}
#endif

TEST(HuffmanTest, ManyRandom) {
  int num_iters = 100;
  srand(0);
  for (int k = 0; k < num_iters; ++k) {
    string raw;
    int len = 1 + rand() % 200;
    for (int i = 0; i < len; ++i) {
      uint8_t ch = 0;
      do {
        // Make a biased distribution
        ch = (rand() & rand() & rand()) & 0xff;
      } while (!std::isprint(ch));
      raw.push_back(ch);
    }
    string compressed = huffman::Compress(raw);
    string decompressed = huffman::Decompress(compressed);
    EXPECT_EQ(raw, decompressed) << "k = " << k;
  }
}
