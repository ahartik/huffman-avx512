#include "codec/huffman.h"

#include <string>
#include <cstdlib>
#include <cstdint>

#include "benchmark/benchmark.h""

static void BM_CompressBiased(benchmark::State& state) {
  std::string raw;
  int len = 3000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
    // raw.push_back(uint8_t(rand()));
  }

  for (auto _ : state) {
    std::string compressed = huffman::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

static void BM_CompressUniform(benchmark::State& state) {
  std::string raw;
  int len = 3000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand()));
  }

  for (auto _ : state) {
    std::string compressed = huffman::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

static void BM_DecompressBiased(benchmark::State& state) {
  std::string raw;
  int len = 3000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
    // raw.push_back(uint8_t(rand()));
  }
  std::string compressed = huffman::Compress(raw);
  for (auto _ : state) {
    std::string decompressed = huffman::Decompress(compressed);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

static void BM_DecompressUniform(benchmark::State& state) {
  std::string raw;
  int len = 3000;

  for (int i = 0; i < len; ++i) {
    raw.push_back(uint8_t(rand()));
  }

  std::string compressed = huffman::Compress(raw);
  for (auto _ : state) {
    std::string decompressed = huffman::Decompress(compressed);
  }
  state.SetBytesProcessed(state.iterations() * len);
}

BENCHMARK(BM_CompressBiased);
BENCHMARK(BM_CompressUniform);
BENCHMARK(BM_DecompressBiased);
BENCHMARK(BM_DecompressUniform);
