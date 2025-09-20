#include "codec/huffman.h"
#include "codec/huff0.h"

#include <random>

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <string>

#include <cmath>

#include "benchmark/benchmark.h"

constexpr size_t LEN = 127 << 10;

namespace {

// Generate data corresponding to exponential "Proba" distributions
// used by FSE/Huff0 benchmarks.
//
// Adapted from FiniteStateEntropy/programs/fullbench.c function BMK_genData().
std::string GenerateProbaData(double p, size_t len = LEN) {
  std::string data(len, 0);
  std::mt19937_64 mt;
  std::uniform_real_distribution<> dist(0.0, 1.0);
  double logp = log(1 - p);
  for (char& c : data) {
    c = int(log(dist(mt)) / logp) % 256;
  }
  return data;
}

std::string file_path = "";
std::string_view FileData() {
  static std::string file_data = [] {
    if (file_path.empty()) {
      std::cerr << "No file path given.\n";
      return std::string("");
    }
    std::ifstream fin(file_path);
    if (!fin.good()) {
      std::cout << "Failed to open '" << file_path << "': "
        << strerror(errno) <<"\n";
      abort();
    }
    // fin.exceptions(fin.exceptions() | std::ios::failbit);
        
    std::string data(LEN, 0);
    fin.read(data.data(), LEN);
    data.resize(fin.gcount());
    return data;
  }();
  return file_data;
}

template <typename Compressor>
void BM_CompressBiased(benchmark::State& state) {
#if 0
  srand(0);
  std::string raw;
  for (int i = 0; i < LEN; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
  }
#endif
  std::string raw = GenerateProbaData(0.2);

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
#if 0
  srand(0);
  std::string raw;
  for (int i = 0; i < LEN; ++i) {
    raw.push_back(uint8_t(rand() & rand() & rand()));
  }
#else
  std::string raw = GenerateProbaData(0.2);
#endif

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

template <typename Compressor>
void BM_CompressFile(benchmark::State& state) {
  std::string_view raw = FileData();
  if (raw.empty()) {
    state.SkipWithMessage("No file path specified");
    return;
  }

  for (auto _ : state) {
    std::string compressed = Compressor::Compress(raw);
  }
  state.SetBytesProcessed(state.iterations() * raw.size());
  state.SetLabel(file_path);
}

template <typename Compressor>
void BM_DecompressFile(benchmark::State& state) {
  std::string_view raw = FileData();
  if (raw.empty()) {
    state.SkipWithMessage("No file path specified");
    return;
  }

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
  BENCHMARK(BM_CompressFile<TYPE>);   \
  BENCHMARK(BM_DecompressFile<TYPE>);   \

DEFINE_BENCHMARKS(::huffman::HuffmanCompressor)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<1>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<4>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorMulti<8>)

DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<8>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<16>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<24>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<32>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<40>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvx<48>)

DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxGather<8>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxGather<16>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxGather<24>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxGather<32>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxGather<40>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxGather<48>)

DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxPermute<8>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxPermute<16>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxPermute<24>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxPermute<32>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxPermute<40>)
DEFINE_BENCHMARKS(::huffman::HuffmanCompressorAvxPermute<48>)

DEFINE_BENCHMARKS(::huffman::Huff0Compressor)

// Custom 
int main(int argc, char** argv)
{
  // std::cout << "argc = " << argc << "\n";
  // std::cout << "argv[1] = " << argv[1] << "\n";
  ::benchmark::MaybeReenterWithoutASLR(argc, argv);                     \
  ::benchmark::Initialize(&argc, argv);
  // ::benchmark::Initialize modifies (argc, argv) to contain only the leftover
  // arguments.
  if (argc > 1) {
    ::file_path = argv[1];
  }

  ::benchmark::RunSpecifiedBenchmarks();
}
