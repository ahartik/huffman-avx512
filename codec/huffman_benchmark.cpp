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
  double ratio = double(compressed.size()) / raw.size();
  state.counters["ratio"] = ratio;
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
  double ratio = double(compressed.size()) / raw.size();
  state.counters["ratio"] = ratio;
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
  double ratio = double(compressed.size()) / raw.size();
  state.counters["ratio"] = ratio;
}

const char* const LOREM = R"(
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
occaecat cupidatat non proident, sunt in culpa qui officia deserunt
mollit anim id est laborum.
    )";

template <typename Compressor>
void BM_CompressLorem(benchmark::State& state) {
  std::string raw;
  while (raw.size() < LEN) {
    raw.append(LOREM);
  }
  raw.resize(LEN);

  for (auto _ : state) {
    std::string compressed = Compressor::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * raw.size());
}

template <typename Compressor>
void BM_DecompressLorem(benchmark::State& state) {
  std::string raw;
  while (raw.size() < LEN) {
    raw.append(LOREM);
  }
  raw.resize(LEN);

  std::string compressed = Compressor::Compress(raw);
  for (auto _ : state) {
    std::string decompressed = Compressor::Decompress(compressed);
  }
  state.SetBytesProcessed(state.iterations() * raw.size());
  double ratio = double(compressed.size()) / raw.size();
  state.counters["ratio"] = ratio;
}

}  // namespace

#define DEFINE_BENCHMARKS(TYPE)          \
  BENCHMARK(BM_CompressBiased<TYPE>);    \
  BENCHMARK(BM_CompressUniform<TYPE>);   \
  BENCHMARK(BM_CompressShort<TYPE>);     \
  BENCHMARK(BM_CompressLorem<TYPE>);     \
  BENCHMARK(BM_DecompressBiased<TYPE>);  \
  BENCHMARK(BM_DecompressUniform<TYPE>); \
  BENCHMARK(BM_DecompressShort<TYPE>);   \
  BENCHMARK(BM_DecompressLorem<TYPE>);   \

DEFINE_BENCHMARKS(::huffman::HuffmanCompressor)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<4>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<8>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<8>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<16>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<32>)

DEFINE_BENCHMARKS(::huffman::Huff0Compressor)
