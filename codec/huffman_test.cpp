#include "codec/huffman.h"

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
      huffman::HuffmanCompressorMulti<2>,
      huffman::HuffmanCompressorMulti<4>
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
}

TYPED_TEST(CompressorTest, LongCodes) {
  string text;
  for (int i = 0; i < 20; ++i) {
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
