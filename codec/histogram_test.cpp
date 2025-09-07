#include "codec/histogram.h"

#include <cstdlib>

#include <string>

#include "gtest/gtest.h"

using huffman::ByteHistogram;

typedef ByteHistogram (*HistogramFunction)(std::string_view);

class HistogramTest : public testing::TestWithParam<HistogramFunction> {
 public:
  ByteHistogram make(std::string_view str) { return GetParam()(str); }
};

TEST_P(HistogramTest, ShortSanity) {
  ByteHistogram hist = make("foobar");
  EXPECT_EQ(hist['f'], 1);
  EXPECT_EQ(hist['o'], 2);
  EXPECT_EQ(hist['b'], 1);
  EXPECT_EQ(hist['a'], 1);
  EXPECT_EQ(hist['r'], 1);
  EXPECT_EQ(hist['q'], 0);
}

TEST_P(HistogramTest, Long) {
  std::string str;
  int len = 2007;
  for (int i = 0; i < len; ++i) {
    uint8_t ch = (rand() & rand() & rand()) & 0xff;
    str.push_back(ch);
  }
  auto hist = make(str);
  // Simple code is assumed to be bug free.
  auto simple_hist = huffman::MakeHistogramSimple(str);

  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(hist[i], simple_hist[i]) << "i = " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Main, HistogramTest,
                         testing::Values(&huffman::MakeHistogram));
INSTANTIATE_TEST_SUITE_P(Simple, HistogramTest,
                         testing::Values(&huffman::MakeHistogramSimple));
INSTANTIATE_TEST_SUITE_P(Multi, HistogramTest,
                         testing::Values(&huffman::MakeHistogramMulti));
INSTANTIATE_TEST_SUITE_P(GatherScatter, HistogramTest,
                         testing::Values(&huffman::MakeHistogramGatherScatter));
INSTANTIATE_TEST_SUITE_P(Vectorized, HistogramTest,
                         testing::Values(&huffman::MakeHistogramVectorized));
