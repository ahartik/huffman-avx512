#include "codec/huffman.h"
#include "codec/huff0.h"

#include <cstdlib>
#include <cstdio>

#include <format>
#include <string>
#include "gtest/gtest.h"

using std::string;


template<typename T>
class CompressorTest : public ::testing::Test {
  public:
};

class NameGenerator {
 public:
  template <typename T>
  static std::string GetName(int) {
    return T::name();
  }
};

using Compressors = ::testing::Types<huffman::HuffmanCompressor,
      huffman::HuffmanCompressorMulti<4>,
      huffman::HuffmanCompressorMulti<8>,
      // XXX: Fix the decompressor
      huffman::HuffmanCompressorAvx,
      huffman::Huff0Compressor
      >;
TYPED_TEST_SUITE(CompressorTest, Compressors, NameGenerator);

TYPED_TEST(CompressorTest, Hello) {
  string raw = "Hello World";

  string compressed = TypeParam::Compress(raw);
  std::cout << std::flush;
  for (size_t i = 0; i < compressed.size(); ++i) {
    printf("%02x ", int(uint8_t(compressed[i])));
    if (i % 8 == 7) {
      printf("\n");
    }
  }
  printf("\n");
  std::cout << std::flush;
  string decompressed = TypeParam::Decompress(compressed);
  EXPECT_EQ(raw, decompressed);
}

TYPED_TEST(CompressorTest, LongerText) {
  string raw = R"(
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
occaecat cupidatat non proident, sunt in culpa qui officia deserunt
mollit anim id est laborum.
    )";

  string compressed = TypeParam::Compress(raw);
  std::cout << std::flush;
  for (size_t i = 0; i < compressed.size(); ++i) {
    printf("%02x ", int(uint8_t(compressed[i])));
    if (i % 8 == 7) {
      printf("\n");
    }
  }
  printf("\n");
  std::cout << std::flush;
  string decompressed = TypeParam::Decompress(compressed);
  EXPECT_EQ(raw, decompressed);
}

TYPED_TEST(CompressorTest, LongRandom) {
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
  string compressed = TypeParam::Compress(raw);
  string decompressed = TypeParam::Decompress(compressed);
  EXPECT_EQ(raw, decompressed);
}

TYPED_TEST(CompressorTest, SingleSymbolOnly) {
  string compressed = TypeParam::Compress("AAA");
  string decompressed = TypeParam::Decompress(compressed);
  EXPECT_EQ("AAA", decompressed);

  const string long_a(1000, 'a');
  compressed = TypeParam::Compress(long_a);
  decompressed = TypeParam::Decompress(compressed);
  EXPECT_EQ(long_a, decompressed);
}

TYPED_TEST(CompressorTest, LongCodes) {
  const int kLogSize = 16;
  string text;
  for (int i = 0; i < kLogSize; ++i) {
    for (int j = 0; j < (1<<i); ++j) {
      text.push_back('A' + i);
    }
  }
  string compressed = TypeParam::Compress(text);
  string decompressed = TypeParam::Decompress(compressed);
  EXPECT_EQ(text, decompressed);
}

#if 1
TYPED_TEST(CompressorTest, EmptyString) {
  string compressed = TypeParam::Compress("");
  string decompressed = TypeParam::Decompress(compressed);
  EXPECT_EQ("", decompressed);
}
#endif

TYPED_TEST(CompressorTest, ManyRandom) {
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
        ch ^= 'A';
      } while (!std::isprint(ch));
      raw.push_back(ch);
    }
    string compressed = TypeParam::Compress(raw);
    string decompressed = TypeParam::Decompress(compressed);
    // Stop at first failure
    ASSERT_EQ(raw, decompressed) << "k = " << k;
  }
}


TEST(MultiTest, Compress2) {
  string raw = "Hello World";
  string compressed = huffman::CompressMulti<2>(raw);
  std::cout << std::flush;
  for (size_t i = 0; i < compressed.size(); ++i) {
    printf("%02x ", int(uint8_t(compressed[i])));
    if (i % 8 == 7) {
      printf("\n");
    }
  }
  printf("\n");
}

TEST(InternalsTest, CountSymbols) {
  std::string text = "foofoobarbar";
  int sym_count[256];
  ::huffman::internal::CountSymbols(text, sym_count);
}
