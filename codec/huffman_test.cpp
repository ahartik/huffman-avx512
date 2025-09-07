#include "codec/huff0.h"
#include "codec/huffman.h"

#include <cstdio>
#include <cstdlib>
#include <format>
#include <random>

#include <format>
#include <string>
#include "gtest/gtest.h"

using std::string;

template <int K>
class AvxCheckCompressor {
 public:
  static std::string Compress(std::string_view raw) {
    std::string avx = huffman::CompressMultiAvx512<K>(raw);
    std::string regular = huffman::CompressMulti<K>(raw);
    EXPECT_EQ(avx, regular);
    return avx;
  }
  static std::string Decompress(std::string_view compressed) {
    std::string avx = huffman::DecompressMultiAvx512<K>(compressed);
    std::string regular = huffman::DecompressMulti<K>(compressed);
    EXPECT_EQ(avx, regular);
    return avx;
  }

  static std::string name() { return std::format("AvxCheckCompressor<{}>", K); }
};

template <typename T>
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

using Compressors = ::testing::Types<
    huffman::HuffmanCompressor, huffman::HuffmanCompressorMulti<4>,
    huffman::HuffmanCompressorMulti<8>, huffman::HuffmanCompressorMulti<32>,
    // XXX: Fix the decompressor
    huffman::HuffmanCompressorAvx<8>, huffman::HuffmanCompressorAvx<32>,
    AvxCheckCompressor<8>, huffman::Huff0Compressor>;
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
TYPED_TEST(CompressorTest, EqualCounts) {
  // Produce a string that is not compressible due to containing an equal
  // amount of all possible bytes. This should result in 8 bits per symbol.
  //
  // This tests some edge cases in the code, which caused a bug previously.
  string raw;
  for (int i = 0; i < 4; ++i)
    for (int c = 0; c < 256; ++c)
      raw.push_back(c);
  std::shuffle(raw.begin(), raw.end(), std::mt19937());

  string compressed = TypeParam::Compress(raw);
  string decompressed = TypeParam::Decompress(compressed);
  EXPECT_EQ(raw, decompressed);
}

TYPED_TEST(CompressorTest, LongRandom) {
  string raw;
  srand(0);
  int len = 100'000;
  for (int i = 0; i < len; ++i) {
    uint8_t ch = 0;
    do {
      // Make a biased distribution
      ch = (rand() & rand() & rand()) & 0xff;
    // } while (!std::isprint(ch));
    } while (false);
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
    for (int j = 0; j < (1 << i); ++j) {
      text.push_back('A' + i);
    }
  }
  std::shuffle(text.begin(), text.end(), std::mt19937());
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
  std::mt19937 mt;
  int num_iters = 100;
  for (int k = 0; k < num_iters; ++k) {
    string raw;
    int len = 1 + mt() % 1000;
    for (int i = 0; i < len; ++i) {
      uint8_t ch = 0;
      do {
        // Make a biased distribution
        ch = (mt() & mt()) & 0xff;
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
