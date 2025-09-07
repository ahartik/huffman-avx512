#include "codec/histogram.h"

#include <cstdlib>

#include <string>
#include <random>
#include "benchmark/benchmark.h"

using namespace ::huffman;

namespace {

constexpr int LEN = 100'000 / 32;
typedef ByteHistogram (*HistogramFunction)(std::string_view);

template<HistogramFunction Func>
void BM_Uniform(benchmark::State& state) {
  std::string text;
  for (int i = 0; i < LEN; ++i) {
    text.push_back(rand() % 256);
  }
  for (auto _ : state) {
    auto hist = Func(text);
    ::benchmark::DoNotOptimize(hist[7]);
  }
  state.SetBytesProcessed(state.iterations() * text.size());
}

template<HistogramFunction Func>
void BM_Biased(benchmark::State& state) {
  srand(0);
  const int kLogSize = 18;
  std::string text;
  for (int i = 0; i < kLogSize; ++i) {
    for (int j = 0; j < (1 << i); ++j) {
      text.push_back('A' + i);
    }
  }
  std::shuffle(text.begin(), text.end(), std::mt19937());
  text = text.substr(0, LEN);
  for (auto _ : state) {
    auto hist = Func(text);
    ::benchmark::DoNotOptimize(hist[7]);
  }
  state.SetBytesProcessed(state.iterations() * text.size());
}

}  // namespace


#define DEFINE_BENCHMARKS(FUNC) \
  BENCHMARK_TEMPLATE(BM_Uniform, FUNC); \
  BENCHMARK_TEMPLATE(BM_Biased, FUNC);


DEFINE_BENCHMARKS(MakeHistogram);
DEFINE_BENCHMARKS(MakeHistogramSimple);
DEFINE_BENCHMARKS(MakeHistogramMulti);
DEFINE_BENCHMARKS(MakeHistogramVectorized);
DEFINE_BENCHMARKS(MakeHistogramGatherScatter);
