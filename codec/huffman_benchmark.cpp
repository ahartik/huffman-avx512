#include "codec/huffman.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#include "benchmark/benchmark.h"

namespace {
template <typename Compressor>
void BM_CompressBiased(benchmark::State& state) {
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
  std::string raw;
  int len = 100000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
  }

  std::string compressed = Compressor::Compress(raw);
  for (auto _ : state) {
    std::string decompressed = Compressor::Decompress(compressed);
  }
  state.SetBytesProcessed(state.iterations() * len);
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
