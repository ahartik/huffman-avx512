#include "codec/huffman.h"
#include "codec/huff0.h"

#include <iostream>

#include <cstdint>
#include <cstdlib>

#include <string>
#include <random>

#include "benchmark/benchmark.h"

constexpr int LEN = 100'000;

namespace {
template <typename Compressor>
void BM_CompressBiased(benchmark::State& state) {
  srand(0);
  std::string raw;
  for (int i = 0; i < LEN; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
  }

  for (auto _ : state) {
    std::string compressed = Compressor::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * raw.size());
}

template <typename Compressor>
void BM_CompressUniform(benchmark::State& state) {
  srand(0);
  std::string raw;
  int len = LEN;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand()));
  }

  for (auto _ : state) {
    std::string compressed = Compressor::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * raw.size());
}

template <typename Compressor>
void BM_DecompressBiased(benchmark::State& state) {
  srand(0);
  std::string raw;

  for (int i = 0; i < LEN; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
  }
  std::string compressed = Compressor::Compress(raw);
  for (auto _ : state) {
    std::string decompressed = Compressor::Decompress(compressed);
  }
  state.SetBytesProcessed(state.iterations() * raw.size());
}

template <typename Compressor>
void BM_DecompressUniform(benchmark::State& state) {
  srand(0);
  std::string raw;

  for (int i = 0; i < LEN; ++i) {
    raw.push_back(uint8_t(rand()));
  }

  std::string compressed = Compressor::Compress(raw);
  for (auto _ : state) {
    std::string decompressed = Compressor::Decompress(compressed);
  }
  state.SetBytesProcessed(state.iterations() * raw.size());
}

template <typename Compressor>
void BM_CompressShort(benchmark::State& state) {
  srand(0);
  std::string raw;
  int len = 100;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
  }

  for (auto _ : state) {
    std::string compressed = Compressor::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

template <typename Compressor>
void BM_DecompressShort(benchmark::State& state) {
  srand(0);
  std::string raw;
  int len = 100;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
  }

  std::string compressed = Compressor::Compress(raw);
  for (auto _ : state) {
    std::string decompressed = Compressor::Decompress(compressed);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

template <typename Compressor>
void BM_CompressLong(benchmark::State& state) {
  srand(0);
  std::string raw;
  int len = 100000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
  }

  for (auto _ : state) {
    std::string compressed = Compressor::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

template <typename Compressor>
void BM_DecompressLong(benchmark::State& state) {
  // std::cout << "BM_DecompressLong\n";
  std::mt19937 mt;
  std::string raw;
  int len = 100000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(mt() & mt() & mt()));
  }

  std::string compressed = Compressor::Compress(raw);
  int64_t total_raw = 0;
  for (auto _ : state) {
    std::string decompressed = Compressor::Decompress(compressed);
    total_raw += decompressed.size();
  }
  state.SetBytesProcessed(total_raw);

  double ratio = double(compressed.size()) / raw.size();
  state.counters["ratio"] = ratio;
}

void BM_CountSymbolsBiased(benchmark::State& state) {
  srand(0);
  const int kLogSize = 16;
  std::string text;
  for (int i = 0; i < kLogSize; ++i) {
    for (int j = 0; j < (1 << i); ++j) {
      text.push_back('A' + i);
    }
  }
  std::shuffle(text.begin(), text.end(), std::mt19937());
  text = text.substr(0, 100000 / 32);
  for (auto _ : state) {
    int sym_count[256] = {};
    huffman::internal::CountSymbols(text, sym_count);
  }
  state.SetBytesProcessed(state.iterations() * text.size());
}
void BM_CountSymbolsUniform(benchmark::State& state) {
  const int len = 100000 / 32;
  std::string text;
  for (int i = 0; i < len; ++i) {
    text.push_back(rand() % 256);
  }
  for (auto _ : state) {
    int sym_count[256] = {};
    huffman::internal::CountSymbols(text, sym_count);
  }
  state.SetBytesProcessed(state.iterations() * text.size());
}

void BM_CountSymbolsLongBiased(benchmark::State& state) {
  const int kLogSize = 16;
  std::string text;
  for (int i = 0; i < kLogSize; ++i) {
    for (int j = 0; j < (1 << i); ++j) {
      text.push_back('A' + i);
    }
  }
  std::shuffle(text.begin(), text.end(), std::mt19937());
  text = text.substr(0, 100000);

  for (auto _ : state) {
    int sym_count[256] = {};
    huffman::internal::CountSymbols(text, sym_count);
  }
  state.SetBytesProcessed(state.iterations() * text.size());
}

}  // namespace

#define DEFINE_BENCHMARKS(TYPE)          \
  BENCHMARK(BM_CompressBiased<TYPE>);    \
  BENCHMARK(BM_CompressUniform<TYPE>);   \
  BENCHMARK(BM_CompressShort<TYPE>);     \
  BENCHMARK(BM_CompressLong<TYPE>);      \
  BENCHMARK(BM_DecompressBiased<TYPE>);  \
  BENCHMARK(BM_DecompressUniform<TYPE>); \
  BENCHMARK(BM_DecompressShort<TYPE>);   \
  BENCHMARK(BM_DecompressLong<TYPE>);

DEFINE_BENCHMARKS(::huffman::HuffmanCompressor)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<4>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<8>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<8>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<16>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<32>)

DEFINE_BENCHMARKS(::huffman::Huff0Compressor)


BENCHMARK(BM_CountSymbolsUniform);
BENCHMARK(BM_CountSymbolsBiased);
BENCHMARK(BM_CountSymbolsLongBiased);
