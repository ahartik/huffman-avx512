#include "codec/huffman.h"
#include "codec/huff0.h"

#include <iostream>

#include <cstdint>
#include <cstdlib>

#include <string>
#include <random>

#include "benchmark/benchmark.h"

namespace {
template <typename Compressor>
void BM_CompressBiased(benchmark::State& state) {
  srand(0);
  std::string raw;
  int len = 3000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
  }

  for (auto _ : state) {
    std::string compressed = Compressor::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

template <typename Compressor>
void BM_CompressUniform(benchmark::State& state) {
  srand(0);
  std::string raw;
  int len = 3000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand()));
  }

  for (auto _ : state) {
    std::string compressed = Compressor::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

template <typename Compressor>
void BM_DecompressBiased(benchmark::State& state) {
  srand(0);
  std::string raw;
  int len = 3000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
    // raw.push_back(uint8_t(rand()));
  }
  std::string compressed = Compressor::Compress(raw);
  for (auto _ : state) {
    std::string decompressed = Compressor::Decompress(compressed);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

template <typename Compressor>
void BM_DecompressUniform(benchmark::State& state) {
  srand(0);
  std::string raw;
  int len = 3000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand()));
  }

  std::string compressed = Compressor::Compress(raw);
  for (auto _ : state) {
    std::string decompressed = Compressor::Decompress(compressed);
  }
  state.SetBytesProcessed(state.iterations() * len);
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
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<2>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<4>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<8>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<8>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<16>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<32>)

DEFINE_BENCHMARKS(::huffman::Huff0Compressor)
