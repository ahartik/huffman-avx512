#include "codec/huffman.h"

#include <arpa/inet.h>
#include <immintrin.h>
#include <x86intrin.h>

#include <cassert>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <type_traits>
#include <vector>

namespace huffman {

namespace {

// Maximum code length we want to use.  Shorter max code lengths makes for
// faster compression and decompression.
//
// Max code length 13 observes some cache misses with AVX-512 decompression,
// reducing performance slightly and increasing variance. Lower max code length
// is used to stay comparable to Huff0, which defaults to 11.
const int kMaxCodeLength = 12;
// const uint32_t kMaxCodeMask = (1 << kMaxCodeLength) - 1;
// Maximum code length that would be optimal in terms of compression.  We use
// shorter codes with slightly worse compression to gain better performance.
const int kMaxOptimalCodeLength = 32;

#ifdef HUFF_DEBUG
#define HUFF_VLOG 1
#elif !defined(HUFF_VLOG)
#define HUFF_VLOG 0
#endif

#define UNROLL8 _Pragma("GCC unroll 8")

#define DLOG(level) \
  if (level <= HUFF_VLOG) std::cout

void MyAbort() {
  std::cout << std::flush;
  std::cerr << std::flush;
  abort();
}

bool VecEqual(__m512i a, __m512i b) {
  __mmask8 k = _mm512_cmpeq_epi64_mask(a, b);
  uint32_t mask_i = _cvtmask8_u32(k);
  // std::cout << "mask_i = " << mask_i << "\n";
  return mask_i == 0xff;
}

std::string Int64VecToString(__m512i vec) {
  uint64_t nums[8];
  _mm512_storeu_epi64(nums, vec);
  std::ostringstream s;
  s << "[";
  for (int i = 0; i < 8; ++i) {
    s << std::format("{:016x}", nums[i]);
    if (i != 7) {
      s << ", ";
    }
  }
  s << "]";
  return s.str();
}

#define ASSERT_VEC_EQ(a, b)                                             \
  do {                                                                  \
    if (!VecEqual(a, b)) {                                              \
      std::cout << "" #a " != " #b " :\n";                              \
      std::cout << Int64VecToString(a) << " != " << Int64VecToString(b) \
                << "\n";                                                \
      MyAbort();                                                        \
    }                                                                   \
  } while (0)

}  // namespace

namespace internal {

void CountSymbolsVectorized(std::string_view text, int* sym_count) {
  // I think the () at the end of the new-expression guarantees that the array
  // gets zeroed.
  const std::unique_ptr<std::array<uint32_t, 256>[]> tmp_count(
      new std::array<uint32_t, 256>[16]());
  assert(tmp_count[2][7] == 0);
  const size_t text_size = text.size();
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(text.data());
  const uint8_t* end = ptr + text_size;
  if (ptr + 16 < end) {
    __m128i nextVec = _mm_loadu_si128((const __m128i*)ptr);
    ptr += 16;
    while (ptr + 16 < end) {
      __m128i vec = nextVec;
      nextVec = _mm_loadu_si128((const __m128i*)ptr);
      ptr += 16;

      // TODO: This can also be achieved using C++ templates.
#if 1
#define ADD_ONE(j)                         \
  do {                                     \
    uint64_t b = _mm_extract_epi8(vec, j); \
    ++tmp_count[j][b];                     \
  } while (0)

      // clang-format off
      ADD_ONE(0); ADD_ONE(1); ADD_ONE(2); ADD_ONE(3);
      ADD_ONE(4); ADD_ONE(5); ADD_ONE(6); ADD_ONE(7);
      ADD_ONE(8); ADD_ONE(9); ADD_ONE(10); ADD_ONE(11);
      ADD_ONE(12); ADD_ONE(13); ADD_ONE(14); ADD_ONE(15);
      // clang-format on
#undef ADD_ONE
#elif 0

#define ADD_TWO(j)                        \
  do {                                     \
    uint16_t b = _mm_extract_epi16(vec, j);\
    ++tmp_count[j*2][b & 0xff];            \
    ++tmp_count[j*2+1][(b>>8) & 0xff];     \
  } while (0)

      // clang-format off
      ADD_TWO(0); ADD_TWO(1); ADD_TWO(2); ADD_TWO(3);
      ADD_TWO(4); ADD_TWO(5); ADD_TWO(6); ADD_TWO(7);
      // clang-format on
#undef ADD_TWO

#else

#define ADD_FOUR(j)                        \
  do {                                     \
    uint32_t b = _mm_extract_epi32(vec, j);\
    ++tmp_count[j*4][b & 0xff];            \
    ++tmp_count[j*4+1][(b>>8) & 0xff];     \
    ++tmp_count[j*4+2][(b>>16) & 0xff];     \
    ++tmp_count[j*4+3][(b>>24) & 0xff];     \
  } while (0)
      // clang-format off
      ADD_FOUR(0); ADD_FOUR(1); ADD_FOUR(2); ADD_FOUR(3);
      // clang-format on
#undef ADD_FOUR
#endif
    }
    ptr -= 16;
  }

  while (ptr < end) {
    ++tmp_count[0][*ptr++];
  }
  // TODO: Vectorize this
  for (int c = 0; c < 256; ++c) {
    sym_count[c] = 0;
    UNROLL8 for (int j = 0; j < 16; ++j) { sym_count[c] += tmp_count[j][c]; }
  }
}

// This is not very fast :(
void CountSymbolsGatherScatter(std::string_view text, int* sym_count) {
  // I think the () at the end of the new-expression guarantees that the array
  // gets zeroed.
  const std::unique_ptr<uint32_t[]> tmp_count(new uint32_t[16 * 256]());
  assert(tmp_count[27] == 0);
  const size_t text_size = text.size();
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(text.data());
  const uint8_t* end = ptr + text_size;

  uint32_t index_offset[16];
  for (int i = 0; i < 16; ++i) {
    index_offset[i] = 256 * i;
  }
  const __m512i offset_v = _mm512_loadu_epi16(index_offset);
  const __m512i one32 = _mm512_set1_epi32(1);
  while (ptr + 16 < end) {
    __m128i bytes = _mm_loadu_si128((const __m128i*)ptr);
    ptr += 16;
    __m512i index = _mm512_cvtepu8_epi32(bytes);
    index = _mm512_add_epi32(index, offset_v);

    __m512i cnt = _mm512_i32gather_epi32(index, tmp_count.get(), 4);
    cnt = _mm512_add_epi32(cnt, one32);

    _mm512_i32scatter_epi32(tmp_count.get(), index, cnt, 4);
  }

  while (ptr < end) {
    ++tmp_count[*ptr++];
  }
  // TODO: Vectorize this
  for (int c = 0; c < 256; ++c) {
    sym_count[c] = 0;
    UNROLL8 for (int j = 0; j < 16; ++j) { sym_count[c] += tmp_count[j * 256 + c]; }
  }
}

void CountSymbols(std::string_view text, int* sym_count) {
  // Idea copied from Huff0: count in four stripes to maximize superscalar
  if (text.size() < 1500) {
    const size_t text_size = text.size();
    for (size_t i = 0; i < text_size; ++i) {
      ++sym_count[uint8_t(text[i])];
    }
  } else {
    // CountSymbolsVectorized(text, sym_count);
    // CountSymbolsGatherScatter(text, sym_count);
#if 1
    // 4K, hopefully still fits on the stack.
    
    int tmp_count[8][256] = {};
    const size_t text_size = text.size();
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(text.data());
    const uint8_t* end = ptr + text_size;
    // Unrolling here is a very very small improvement
#pragma GCC unroll 4
    while (ptr + 7 < end) {
      // Four bytes at a time is slightly faster than 8.
      uint64_t data;
      memcpy(&data, ptr, 8);
      ptr += 8;
      UNROLL8 for (int j = 0; j < 8; ++j) {
        ++tmp_count[j][(data >> (j * 8)) & 0xff];
      }
    }
    while (ptr < end) {
      ++tmp_count[0][*ptr++];
    }
    for (int c = 0; c < 256; ++c) {
      for (int j = 0; j < 8; ++j) {
        sym_count[c] += tmp_count[j][c];
      }
    }
#endif
  }
}

// Returns the sizes of slices
template <int K>
std::array<size_t, K> SliceSizes(size_t len) {
  std::array<size_t, K> sizes;
  for (int i = 0; i < K; ++i) {
    sizes[i] = len / K;
  }
  for (size_t i = 0; i < len % K; ++i) {
    ++sizes[i];
  }
  return sizes;
}

}  // namespace internal
   //

using namespace huffman::internal;

namespace {

using vec8x64 = __m512i;
using vec64x8 = __m512i;
using mask8 = __mmask8;
using mask64 = __mmask64;

int CountBits(uint64_t x) { return __builtin_popcountll(x); }

inline uint64_t ToBigEndian64(uint64_t x) { return __builtin_bswap64(x); }

inline uint64_t FromBigEndian64(uint64_t x) { return __builtin_bswap64(x); }

inline uint64_t ReverseBits64(uint64_t x) {
  // https://graphics.stanford.edu/%7Eseander/bithacks.html#ReverseParallel

  const uint64_t bits55 = 0x5555555555555555ull;
  const uint64_t bits33 = 0x3333333333333333ull;
  const uint64_t bits0f = 0x0f0f0f0f0f0f0f0full;
  // Swap odd and even bits
  x = ((x >> 1) & bits55) | ((x & bits55) << 1);
  // swap consecutixe pairs
  x = ((x >> 2) & bits33) | ((x & bits33) << 2);
  // // swap nibbles ...
  x = ((x >> 4) & bits0f) | ((x & bits0f) << 4);
  // Bytes can be reversed with this instruction instead.
  return __builtin_bswap64(x);
}

inline uint16_t ReverseBits16(uint16_t x) {
  // https://graphics.stanford.edu/%7Eseander/bithacks.html#ReverseParallel

  const uint16_t bits55 = 0x5555;
  const uint16_t bits33 = 0x3333;
  const uint16_t bits0f = 0x0f0f;
  // Swap odd and even bits
  x = ((x >> 1) & bits55) | ((x & bits55) << 1);
  // swap consecutixe pairs
  x = ((x >> 2) & bits33) | ((x & bits33) << 2);
  // // swap nibbles ...
  x = ((x >> 4) & bits0f) | ((x & bits0f) << 4);
  // Reverse the two bytes:
  return (x >> 8) | (x << 8);
}

vec8x64 GetWord16(vec8x64 vec, int W) {
  switch (W) {
    case 0:
      return _mm512_and_epi64(vec, _mm512_set1_epi64(0xffffULL));
    case 1: {
      vec64x8 ctrl = _mm512_set4_epi64(0xfffffffFffff0b0a, 0xfffffffFffff0302,
                                       0xfffffffFffff0b0a, 0xfffffffFffff0302);
      return _mm512_shuffle_epi8(vec, ctrl);
    }
    case 2: {
      vec64x8 ctrl = _mm512_set4_epi64(0xfffffffFffff0d0c, 0xfffffffFffff0504,
                                       0xfffffffFffff0d0c, 0xfffffffFffff0504);
      return _mm512_shuffle_epi8(vec, ctrl);
    }
    case 3: {
      return _mm512_srli_epi64(vec, 48);
    }
    default:
      // This should not be reached.
      assert(false);
      return vec;
  }
}

vec8x64 GetWord16ToTopBits(vec8x64 vec, int W) {
  switch (W) {
    case 0:
      return _mm512_slli_epi64(vec, 48);
    case 1: {
#if 1
      vec64x8 ctrl =
          _mm512_set4_epi64(0x0b0a'FFFF'FFFF'FFFF, 0x0302'ffff'fffF'ffff,
                            0x0b0a'FFFF'FFFF'FFFF, 0x0302'ffff'fffF'ffff);
      return _mm512_shuffle_epi8(vec, ctrl);
#else
      // Using an extra instruction but fewer constants is a very tiny
      // optimization for our code in particular.
      return _mm512_and_epi64(_mm512_slli_epi64(vec, 32),
                              _mm512_set1_epi64(0xffffULL << 48));

#endif
    }
    case 2: {
#if 1
      vec64x8 ctrl =
          _mm512_set4_epi64(0x0d0c'ffff'fffF'ffff, 0x0504'ffff'fffF'ffff,
                            0x0d0c'ffff'fffF'ffff, 0x0504'ffff'fffF'ffff);
      return _mm512_shuffle_epi8(vec, ctrl);
#else
      // Using an extra instruction but fewer constants is a very tiny
      // optimization for our code in particular.
      return _mm512_and_epi64(_mm512_slli_epi64(vec, 16),
                              _mm512_set1_epi64(0xffffULL << 48));
#endif
    }
    case 3: {
      return _mm512_and_epi64(vec, _mm512_set1_epi64(0xffffULL << 48));
    }
    default:
      // This should not be reached.
      assert(false);
      return vec;
  }
}

struct BitCode {
  BitCode(uint16_t b = 0, uint16_t l = 0) : bits(b), len(l) {}
  uint16_t bits;
  // uint16_t mask;
  uint16_t len;
  // int16_t pad;
  //
};

std::string SymToStr(uint8_t sym) {
  if (std::isprint(sym)) {
    return std::format("{:c}", sym);
  } else {
    return std::format("\\x{:02x}", sym);
  }
}

std::ostream& operator<<(std::ostream& out, const BitCode& code) {
  out << "[";
  for (int i = 0; i < code.len; ++i) {
    out << ((code.bits >> (kMaxCodeLength - 1 - i)) & 1);
  }
  return out << "]";
}

struct Node {
  int count = 0;
  int children[2] = {};
  uint8_t sym = 0;
  // Opposite order, since C++ heap is a max heap and we want
  // to pop smallest counts first.
  bool operator<(const Node& o) const { return count > o.count; }
};

void get_code_len(const std::vector<Node>& tree, const Node* node, int len,
                  uint8_t* len_count) {
  if (node->children[0] == node->children[1]) {
    ++len_count[len];
  } else {
    get_code_len(tree, &tree[node->children[0]], len + 1, len_count);
    get_code_len(tree, &tree[node->children[1]], len + 1, len_count);
  }
}

void write_u32(std::string& out, uint32_t x) {
  static_assert(std::endian::native == std::endian::little,
                "Big endian support not included, but easy to add");
  char bytes[4];
  std::memcpy(&bytes, &x, 4);
  out.append(bytes, 4);
}

uint32_t read_u32(std::string_view& in) {
  uint32_t x;
  std::memcpy(&x, in.data(), 4);
  in.remove_prefix(4);
  return x;
}

// Iterates all symbols in [syms, syms+num_syms) and calls `func` for each
// code. `func` should have signature bool(uint8_t sym, BitCode code).
// If `func` returns false, the iteration is stopped.
template <typename Func>
void ForallCodes(const uint8_t* len_count, const uint8_t* syms, int num_syms,
                 Func func) {
  uint32_t current_code = 0;
  uint64_t current_inc = 1ull << kMaxCodeLength;
  bool aborted = false;

  int i = 0;
  for (int len = 0; (len <= kMaxCodeLength) && !aborted; ++len) {
    for (int j = 0; j < len_count[len]; ++j) {
      if (!func(syms[i], BitCode(current_code, len))) {
        aborted = true;
        break;
      }
      ++i;
      current_code += current_inc;
    }
    current_inc >>= 1;
  }
  // Should have exactly wrapped around:
  if (num_syms != 0 && !aborted) {
    assert(i == num_syms);
    assert(current_code == (1 << kMaxCodeLength));
  }
}

struct CanonicalCoding {
  BitCode codes[256];
  uint8_t sorted_syms[256] = {};
  int num_syms = 0;
  uint8_t len_count[kMaxOptimalCodeLength + 1] = {};
  uint32_t len_mask = 0;
};

// Tweak code lens to reduce max length to kMaxCodeLength.
// This uses the "MiniZ" method as described in
// https://create.stephan-brumme.com/length-limited-prefix-codes/#miniz
void LimitCodeLengths(uint8_t* len_count) {
  bool adjustment_required = 0;
  for (int i = kMaxCodeLength + 1; i <= kMaxOptimalCodeLength; ++i) {
    adjustment_required |= len_count[i] > 0;
    len_count[kMaxCodeLength] += len_count[i];
    len_count[i] = 0;
  }
  uint32_t kraft_total = 0;
  for (int i = 0; i <= kMaxCodeLength; ++i) {
    kraft_total += (len_count[i] << (kMaxCodeLength - i));
  }
  const uint32_t one = 1 << kMaxCodeLength;
  DLOG(1) << "LimitCodeLengths: adjustment_required " << adjustment_required
          << "\n";
  DLOG(1) << "kraft_total: " << kraft_total << " one: " << one << "\n";
  DLOG(1) << std::flush;
  // int second_longest_len = kMaxCodeLength - 1;
  while (kraft_total > one) {
    // Decrease the length of one code with the maximum length.
    --len_count[kMaxCodeLength];
    // Increase the length for some code with currently a shorter length.
    for (int j = kMaxCodeLength - 1; j >= 0; --j) {
      if (len_count[j] > 0) {
        --len_count[j];
        len_count[j + 1] += 2;
        break;
      }
    }
    --kraft_total;
  }
}

CanonicalCoding MakeCanonicalCoding(const int* sym_count) {
  CanonicalCoding coding;

  std::vector<Node> heap;
  std::vector<Node> tree;
  heap.reserve(256);
  // TODO: This too could be optimized, perhaps even using AVX.
  for (int c = 0; c < 256; ++c) {
    if (sym_count[c] != 0) {
      Node node;
      node.count = sym_count[c];
      node.sym = uint8_t(c);
      heap.push_back(node);
      coding.sorted_syms[coding.num_syms] = node.sym;
      ++coding.num_syms;
    }
  }
  if (coding.num_syms == 0) {
    return coding;
  }

  tree.reserve(heap.size());
  std::make_heap(heap.begin(), heap.end());
  while (heap.size() > 1) {
    // Pop two elements
    Node a = heap[0];
    std::pop_heap(heap.begin(), heap.end());
    heap.pop_back();
    Node b = heap[0];
    std::pop_heap(heap.begin(), heap.end());
    heap.pop_back();

    tree.push_back(a);
    tree.push_back(b);

    Node next;
    next.count = a.count + b.count;
    next.children[0] = tree.size() - 2;
    next.children[1] = tree.size() - 1;
    heap.push_back(next);
    std::push_heap(heap.begin(), heap.end());
  }

  get_code_len(tree, &heap[0], 0, coding.len_count);
  // Build "canonical Huffman code".

  // Sort the symbols in decreasing order of frequency.
  std::sort(coding.sorted_syms, coding.sorted_syms + coding.num_syms,
            [&](uint8_t a, uint8_t b) { return sym_count[a] > sym_count[b]; });

  LimitCodeLengths(coding.len_count);

  for (int i = 0; i <= kMaxCodeLength; ++i) {
    if (coding.len_count[i] != 0) {
      coding.len_mask |= 1ull << i;
    }
  }

  ForallCodes(coding.len_count, coding.sorted_syms, coding.num_syms,
              [&coding](uint8_t sym, BitCode code) {
                coding.codes[sym] = code;
                DLOG(1) << SymToStr(sym) << " -> " << code << "\n";
                return true;
              });

  return coding;
}

class CodeWriter {
 public:
  explicit CodeWriter(char* begin, char* end) { Init(begin, end); }

  CodeWriter() {}

  void Init(char* begin, char* end) {
    begin_ = begin;
    end_ = end;
    output_ = end - 8;
    buf_ = 0;
    buf_size_ = 0;
  }

  void WriteCode(BitCode code) {
    WriteFast(code);
    Flush();
  }

  void Flush() {
    DLOG(2) << std::format("Flush(): buf_ = {:016x} buf_size_= {}\n", buf_,
                           buf_size_);
    assert(output_ >= begin_);
    uint32_t num_bytes = buf_size_ >> 3;
    assert(num_bytes <= 8);
    // This assumes little endian:
    memcpy(output_, &buf_, 8);
    output_ -= num_bytes;
    // Leftmost bits were consumed, remaining move to the very left.
    buf_ <<= (buf_size_ & (~7));  // Same as 8 * num_bytes;
    // buf_size_ = 8 * num_bytes;
    buf_size_ &= 7;
  }

  void WriteFast(BitCode code) {
    DLOG(2) << "WriteFast(" << code << "):\n";
    DLOG(2) << std::format("buf_ = {:064B} buf_size_= {}\n", buf_, buf_size_);

    assert(code.len + buf_size_ <= 64);
    assert(code.len >= 0);
    buf_ |= uint64_t(code.bits) << (64 - buf_size_ - kMaxCodeLength);
    buf_size_ += code.len;
  }

  void Finish() {
    char* out_byte = output_ + 7;
    const int bytes_left = (buf_size_ + 7) / 8;
    for (int j = 0; j < bytes_left; ++j) {
      uint8_t top = (buf_ >> 56) & 0xff;
      *out_byte-- = top;
      buf_ <<= 8;
    }
  }

 private:
  char* begin_;
  char* end_;
  char* output_;
  uint64_t buf_;
  uint64_t buf_size_;
};

class CodeReader {
 public:
  // After default construction, the object is not safe to use, but is safe to
  // destruct.
  CodeReader() {}

  CodeReader(const char* begin, const char* end) { Init(begin, end); }

  void Init(const char* begin, const char* end) {
    DLOG(1) << "CodeReader::Init(" << intptr_t(end - begin) << ")\n"
            << std::flush;
    // assert((end == nullptr) || (begin + 8 <= end));
    begin_ = begin;
    input_ = end - 8;
    end_ = end;
    buf_bits_ = 0;
    bits_used_ = 0;
    FillBuffer();
  }

  uint16_t code() const {
    assert(64 - bits_used_ >= kMaxCodeLength);
    // return (buf_bits_ >> (64ll - kMaxCodeLength - bits_used_)) &
    // kMaxCodeMask;
    return (buf_bits_ << bits_used_) >> (64ll - kMaxCodeLength);
  }

  void ConsumeFast(int num_bits) { bits_used_ += num_bits; }

  void ConsumeBits(int num_bits) {
    ConsumeFast(num_bits);
    FillBuffer();
  }

  void FillBuffer() {
    DLOG(2) << "FillBuffer(): bits_used_ = " << bits_used_ << "\n";
    input_ -= bits_used_ >> 3;
    bits_used_ &= 7;
    if (__builtin_expect(input_ < begin_, 0)) {
      // Less than 8 bytes remaining, we simulate a read where the lower
      // address bytes are zero.
      int num_bytes_available = 8 - (begin_ - input_);
      DLOG(2) << "num_bytes_available = " << num_bytes_available << "\n";
      buf_bits_ = 0;
      if (num_bytes_available > 0) {
        memcpy(&buf_bits_, begin_, 8);
        buf_bits_ <<= 8 * (begin_ - input_);
      }
    } else {
      memcpy(&buf_bits_, input_, 8);
    }
    DLOG(2) << "after FillBuffer(): \n";
    DLOG(2) << std::format("buf_bits_: {:064B} bits_used_: {}\n", buf_bits_,
                           bits_used_);
  }

  // Returns false if FillBuffer() should be used instead.
  bool FillBufferFast(bool skip_compare = false) {
    DLOG(2) << "FillBufferFast(): bits_used_ = " << bits_used_ << "\n";
    input_ -= bits_used_ >> 3;
    bits_used_ &= 7;
    if (!skip_compare && __builtin_expect(input_ < begin_, 0)) {
      return false;
    } else {
      memcpy(&buf_bits_, input_, 8);
    }
    DLOG(2) << "after FillBufferFast(): \n";
    DLOG(2) << std::format("buf_bits_: {:064B} bits_used_: {}\n", buf_bits_,
                           bits_used_);
    return true;
  }

  bool is_fast() const { return input_ >= begin_; }

 private:
  const char* input_;
  const char* begin_;
  const char* end_;
  uint64_t buf_bits_;
  uint64_t bits_used_;
};

struct DecodedSym {
  // NOTE: This order is assumed by some AVX code.
  uint8_t code_len = 0;
  uint8_t sym = 0;
};

class Decoder1x {
 public:
  Decoder1x(const uint8_t* len_count, const uint8_t* syms, int num_syms)
      : dtable_((1 << kMaxCodeLength) + 4) {
    DLOG(1) << "Decoder1x:\n";
    ForallCodes(len_count, syms, num_syms, [this](uint8_t sym, BitCode code) {
      uint32_t inc = 1 << (kMaxCodeLength - code.len);
      DecodedSym dsym = {
          .code_len = uint8_t(code.len),
          .sym = sym,
      };
      std::fill(dtable_.begin() + code.bits, dtable_.begin() + code.bits + inc,
                dsym);
      return true;
    });
  }

  // TODO: Two symbols at a time decoding.

  inline int Decode(uint16_t code, uint8_t** out_sym) const {
    DLOG(2) << std::format("Decode({:016B}) \n", code);
    DecodedSym dsym = dtable_[code];

    **out_sym = dsym.sym;
    ++(*out_sym);
    return dsym.code_len;
  }

  inline int Decode1(uint16_t code, uint8_t** out_sym) const {
    return Decode(code, out_sym);
  }

  const DecodedSym* dtable() const { return dtable_.data(); }

  int max_symbols_decoded() const { return 1; }

 private:
  std::vector<DecodedSym> dtable_;
};

struct alignas(4) DecodedSym2x {
  uint8_t num_bits_decoded;
  uint8_t syms[2];
  uint8_t num_syms;
};

class Decoder2x {
 public:
  Decoder2x(const uint8_t* len_count, const uint8_t* syms, int num_syms)
      : single_(len_count, syms, num_syms),
        dtable_(new  // (std::align_val_t(64))
                DecodedSym2x[(1 << kMaxCodeLength) + 4]) {
    ForallCodes(len_count, syms, num_syms, [&](uint8_t sym1, BitCode code1) {
      // Iterate over all codes that are short enough that they can be combined
      // with code1.
      uint32_t last_code = code1.bits;
      ForallCodes(len_count, syms, num_syms, [&](uint8_t sym2, BitCode code2) {
        if (code1.len + code2.len > kMaxCodeLength) {
          // Break iteration, combined code would be too long for our table.
          return false;
        }
        DecodedSym2x msym;
        msym.num_bits_decoded = code1.len + code2.len;
        msym.syms[0] = sym1;
        msym.syms[1] = sym2;
        msym.num_syms = 2;
        const uint32_t code =
            code1.bits | (uint32_t(code2.bits) >> uint32_t(code1.len));
        const uint32_t inc = 1 << (kMaxCodeLength - code1.len - code2.len);

        std::fill(dtable_.get() + code, dtable_.get() + code + inc, msym);
        last_code = code + inc;
        return true;
      });
      // Other codes following code1 must be decoded one symbol at a time.
      DecodedSym2x msym;
      msym.num_bits_decoded = code1.len;
      msym.syms[0] = sym1;
      msym.syms[1] = 0;
      msym.num_syms = 1;
      uint32_t inc = 1 << (kMaxCodeLength - code1.len);
      std::fill(dtable_.get() + last_code, dtable_.get() + code1.bits + inc,
                msym);
      return true;
    });
  }

  int Decode(uint16_t code, uint8_t** output) const {
    auto dsym = dtable_[code];
    (*output)[0] = dsym.syms[0];
    (*output)[1] = dsym.syms[1];
    (*output) += dsym.num_syms;
    return dsym.num_bits_decoded;
  }

  inline int Decode1(uint16_t code, uint8_t** out_sym) const {
    return single_.Decode(code, out_sym);
  }

  const DecodedSym2x* dtable() const { return dtable_.get(); }

  int max_symbols_decoded() const { return 2; }

  using DecodedSymT = DecodedSym2x;

 private:
  Decoder1x single_;
  std::unique_ptr<DecodedSym2x[]> dtable_;
};

}  // namespace

std::string Compress(std::string_view raw) {
  int sym_count[256] = {};
  CountSymbols(raw, sym_count);
  CanonicalCoding coding = MakeCanonicalCoding(sym_count);

  uint64_t output_bits = 0;
  for (int i = 0; i < coding.num_syms; ++i) {
    uint8_t sym = coding.sorted_syms[i];
    output_bits += coding.codes[sym].len * sym_count[sym];
  }

  std::string compressed;
  // TODO: Fail for too long strings.
  write_u32(compressed, raw.size());
  if (raw.size() == 0) {
    // SPECIAL CASE FOR EMPTY STRING:
  }
  write_u32(compressed, coding.len_mask);
  for (uint8_t count : coding.len_count) {
    if (count != 0) {
      compressed.push_back(count);
    }
  }
  compressed.append(reinterpret_cast<char*>(coding.sorted_syms),
                    coding.num_syms);
  const int header_size = compressed.size();
  // TODO: Get rid of slop bytes
  const int kSlop = 8;
  compressed.resize(header_size + (output_bits + 7) / 8 + kSlop);

  CodeWriter writer(&compressed[header_size],
                    compressed.data() + compressed.size());
  const uint8_t* input = reinterpret_cast<const uint8_t*>(raw.data());
  const uint8_t* end = input + raw.size();

  // This pragma showed a 2% speedup once.
  // UNROLL8
  // #pragma GCC unroll 4
  while (input + 3 < end) {
    // We can write multiple codes for each flush.

    UNROLL8 for (int j = 0; j < 4; ++j) {
      BitCode a = coding.codes[*input++];
      writer.WriteFast(a);
    }
    writer.Flush();
  }
  while (input < end) {
    writer.WriteCode(coding.codes[*input++]);
  }
  writer.Finish();
  return compressed;
}

template <typename UsedDecoder>
std::string DecompressImpl(std::string_view compressed) {
  // Build codebook.
  const uint32_t raw_size = read_u32(compressed);
  const uint32_t len_mask = read_u32(compressed);
  uint8_t len_count[32] = {};
  int num_syms = 0;
  for (int i = 0; i < 32; ++i) {
    if (len_mask & (1 << i)) {
      len_count[i] = uint8_t(compressed[0]);
      compressed.remove_prefix(1);
      num_syms += len_count[i];
    }
  }
  if (num_syms == 1) {
    // Output consists of a single symbol only.
    // This causes troubles in decoder, since the symbol is encoded using 0
    // bits. We can just handle this case by itself:
    return std::string(raw_size, compressed[0]);
  }
  UsedDecoder decoder(
      len_count, reinterpret_cast<const uint8_t*>(compressed.data()), num_syms);
  compressed.remove_prefix(num_syms);

  std::string raw(raw_size, 0);
  CodeReader reader(compressed.data(), compressed.data() + compressed.size());

  uint8_t* output = reinterpret_cast<uint8_t*>(raw.data());
  uint8_t* output_end = output + raw_size;
  // Four symbols at a time
  bool readers_good = true;

  const int output_slop = 4 * decoder.max_symbols_decoded() - 1;

  while (readers_good & (output + output_slop < output_end)) {
    UNROLL8 for (int j = 0; j < 4; ++j) {
      int num_bits = decoder.Decode(reader.code(), &output);
      reader.ConsumeFast(num_bits);
    }
    readers_good = reader.FillBufferFast();
  }

  // Last symbols
  while (output != output_end) {
    reader.FillBuffer();
    int bits_read = decoder.Decode1(reader.code(), &output);
    reader.ConsumeBits(bits_read);
  }
  return raw;
}

std::string Decompress(std::string_view compressed) {
  return DecompressImpl<Decoder2x>(compressed);
}

template <int K>
std::string CompressMulti(std::string_view raw) {
  const auto sizes = SliceSizes<K>(raw.size());
  // Start/end input pointers for each part.
  const uint8_t* part_input[K];
  part_input[0] = reinterpret_cast<const uint8_t*>(raw.data());
  for (int i = 1; i < K; ++i) {
    part_input[i] = part_input[i - 1] + sizes[i - 1];
  }
  const uint8_t* part_end[K];
  for (int i = 0; i < K; ++i) {
    part_end[i] = part_input[i] + sizes[i];
  }

  // Count symbols in each component.
  // (I think this performs "value-initialization" on std::array elements,
  // which means the counts are set to zero.)
  std::vector<std::array<int, 256>> part_count(K);
  assert(part_count[0][7] == 0);
  int sym_count[256] = {};
  {
    for (int k = 0; k < K; ++k) {
      CountSymbols(std::string_view(
                       reinterpret_cast<const char*>(part_input[k]), sizes[k]),
                   part_count[k].data());
    }
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < 256; ++i) {
        sym_count[i] += part_count[k][i];
      }
    }
  }
  CanonicalCoding coding = MakeCanonicalCoding(sym_count);

  const int kSlop = 8;
  // Compute starting positions for each part in the output.
  int end_offset[K] = {};
  {
    int pos = 0;
    for (int part = 0; part < K; ++part) {
      int64_t num_bits = 0;
      for (int c = 0; c < 256; ++c) {
        num_bits += part_count[part][c] * coding.codes[c].len;
      }
      pos += sizes[part];
      end_offset[part] = (num_bits + 7) / 8 + kSlop;
    }
  }
  for (int i = 1; i < K; ++i) {
    end_offset[i] += end_offset[i - 1];
  }
  DLOG(1) << "End offsets: ";
  for (int i = 0; i < K; ++i) {
    DLOG(1) << end_offset[i] << " ";
  }
  DLOG(1) << "\n";

  // TODO: Use varints
  const size_t header_size =
      4 + 4 + CountBits(coding.len_mask) + coding.num_syms + (K - 1) * (4);
  const size_t compressed_size = header_size + end_offset[K - 1];
  std::string compressed;
  compressed.reserve(compressed_size);
  write_u32(compressed, raw.size());
  write_u32(compressed, coding.len_mask);
  for (int len = 0; len < 32; ++len) {
    int count = coding.len_count[len];
    if (count != 0) {
      compressed.push_back(count);
    }
  }
  compressed.append(reinterpret_cast<char*>(coding.sorted_syms),
                    coding.num_syms);
  for (int k = 0; k < K - 1; ++k) {
    write_u32(compressed, end_offset[k]);
  }

  assert(compressed.size() == header_size);
  compressed.resize(compressed_size);

  char* part_output[K + 1];
  for (int k = 0; k <= K; ++k) {
    part_output[k] =
        compressed.data() + header_size + ((k == 0) ? 0 : end_offset[k - 1]);
  }

  CodeWriter writer[K];
  for (int k = 0; k < K; ++k) {
    writer[k].Init(part_output[k], part_output[k + 1]);
  }

  // It's slightly strange, but ordering these loops like this is faster.
  // Other way around gets better instruction parallelism, but also has more
  // instructions so ends up slower.
  for (int k = 0; k < K; ++k) {
    while (part_input[k] + 3 < part_end[k]) {
      writer[k].Flush();
      // We can write three codes of up to 14 bits per each flush.
      static_assert(kMaxCodeLength <= 14);
      UNROLL8 for (int j = 0; j < 4; ++j) {
        BitCode a = coding.codes[*part_input[k]++];
        writer[k].WriteFast(a);
      }
    }
  }
  // Write potential last symbols.
  for (int k = 0; k < K; ++k) {
    while (part_input[k] != part_end[k]) {
      BitCode code = coding.codes[*part_input[k]++];
      writer[k].WriteCode(code);
    }
    writer[k].Finish();
  }

  return compressed;
}

template <int K, typename UsedDecoder>
std::string DecompressMultiImpl(std::string_view compressed) {
  const uint32_t raw_size = read_u32(compressed);
  const uint32_t len_mask = read_u32(compressed);
  uint8_t len_count[32] = {};
  int num_syms = 0;
  for (int i = 0; i < 32; ++i) {
    if (len_mask & (1 << i)) {
      len_count[i] = uint8_t(compressed[0]);
      compressed.remove_prefix(1);
      num_syms += len_count[i];
    }
  }

  UsedDecoder decoder(
      len_count, reinterpret_cast<const uint8_t*>(&compressed[0]), num_syms);
  compressed.remove_prefix(num_syms);

  int end_offset[K] = {};
  for (int k = 0; k < K - 1; ++k) {
    end_offset[k] = read_u32(compressed);
  }
  end_offset[K - 1] = compressed.size();

  const auto sizes = SliceSizes<K>(raw_size);
  for (size_t i = 0; i < K; ++i) {
    DLOG(1) << std::format("sizes[{}] = {}\n", i, sizes[i]);
  }
  CodeReader reader[K];
  for (int k = 0; k < K; ++k) {
    // int start_index = (k == 0) ? 0 : end_offset[k - 1];
    // reader[k].Init(compressed.data() + start_index,
    //                compressed.data() + end_offset[k]);
    // This works just as well and seems slightly faster:
    reader[k].Init(compressed.data(), compressed.data() + end_offset[k]);
  }

  std::string raw(raw_size, 0);
  uint8_t* part_output[K];
  uint8_t* part_end[K];
  part_output[0] = reinterpret_cast<uint8_t*>(raw.data());
  for (int i = 1; i < K; ++i) {
    part_output[i] = part_output[i - 1] + sizes[i - 1];
  }
  for (int k = 0; k < K; ++k) {
    part_end[k] = part_output[k] + sizes[k];
  }

  const int output_slop = 4 * decoder.max_symbols_decoded() - 1;
  while (true) {
    bool can_continue = true;
    UNROLL8 for (int k = 0; k < K; ++k) {
      // The checks inside this call could be made faster for k > 0
      // if we were certain that decoding of k > 0 did not go further bcak than
      // for k == 0. This is certain for good inputs, but bad inputs
      // could be crafted where this is not the case. This optimization would
      // improve decompression speed by ~1%.
      can_continue &= reader[k].FillBufferFast();
      // assert(reader[k].is_fast());
      can_continue &= part_output[k] + output_slop < part_end[k];
    }
    if (!can_continue) {
      break;
    }
    UNROLL8 for (int j = 0; j < 4; ++j) {
      UNROLL8 for (int k = 0; k < K; ++k) {
        int code_len = decoder.Decode(reader[k].code(), &part_output[k]);
        reader[k].ConsumeFast(code_len);
      }
    }
  }
  // Read last symbols.
  for (int k = 0; k < K; ++k) {
    reader[k].FillBuffer();
    while (part_output[k] != part_end[k]) {
      const uint64_t code = reader[k].code();
      int bits_read = decoder.Decode1(code, &part_output[k]);
      reader[k].ConsumeBits(bits_read);
      DLOG(1) << "Last sym " << k << " : " << SymToStr(*(part_output[k] - 1))
              << "\n";
    }
  }
  return raw;
}

template <int K>
std::string DecompressMulti(std::string_view compressed) {
  return DecompressMultiImpl<K, Decoder2x>(compressed);
}

// Slowish method for decoding a single stream. Used to finish off decoding one
// character at a time.

template <typename UsedDecoder>
void DecodeSingleStream(const UsedDecoder& decoder,
                        const uint8_t* compressed_begin,
                        const uint8_t* compressed_end, int bit_offset,
                        uint8_t* out_begin, uint8_t* out_end) {
  CodeReader reader(reinterpret_cast<const char*>(compressed_begin),
                    reinterpret_cast<const char*>(compressed_end));
  reader.ConsumeBits(bit_offset);
  uint8_t* output = out_begin;
  while (output <= out_end - decoder.max_symbols_decoded()) {
    int bits = decoder.Decode(reader.code(), &output);
    reader.ConsumeBits(bits);
  }
  while (output < out_end) {
    int bits = decoder.Decode1(reader.code(), &output);
    reader.ConsumeBits(bits);
  }
}

// AVX helper macros

#define DEF_ARR(type, name, val)                             \
  type name[M];                                              \
  do {                                                       \
    UNROLL8 for (int m = 0; m < M; ++m) { name[m] = (val); } \
  } while (0)

#define DEF_VECS(name, val) DEF_ARR(vec8x64, name, val);

#define FORM(m) UNROLL8 for (int m = 0; m < M; ++m)

template <int K>
std::string CompressMultiAvx512(std::string_view raw) {
  // This restriction could be lifted by using masks.
  static_assert(K % 8 == 0);

  const auto sizes = SliceSizes<K>(raw.size());
  uint64_t read_index[K];
  read_index[0] = 0;
  for (int k = 1; k < K; ++k) {
    read_index[k] = read_index[k - 1] + sizes[k - 1];
  }
  uint64_t read_end[K];
  for (int k = 0; k < K; ++k) {
    read_end[k] = read_index[k] + sizes[k];
  }
  // Count symbols in each component.
  // (I think this performs "value-initialization" on std::array elements,
  // which means the counts are set to zero.)
  std::vector<std::array<int, 256>> part_count(K);
  assert(part_count[0][7] == 0);
  int sym_count[256] = {};
  {
    for (int k = 0; k < K; ++k) {
      CountSymbols(raw.substr(read_index[k], sizes[k]), part_count[k].data());
    }
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < 256; ++i) {
        sym_count[i] += part_count[k][i];
      }
    }
  }
  CanonicalCoding coding = MakeCanonicalCoding(sym_count);

  const int kSlop = 8;
  // Compute starting positions for each part in the output.
  uint64_t write_end[K] = {};
  {
    int pos = 0;
    for (int part = 0; part < K; ++part) {
      int64_t num_bits = 0;
      for (int c = 0; c < 256; ++c) {
        num_bits += part_count[part][c] * coding.codes[c].len;
      }
      pos += sizes[part];
      write_end[part] = (num_bits + 7) / 8 + kSlop;
    }
  }
  for (int i = 1; i < K; ++i) {
    write_end[i] += write_end[i - 1];
  }
  DLOG(1) << "End offsets: ";
  for (int i = 0; i < K; ++i) {
    DLOG(1) << write_end[i] << " ";
  }
  DLOG(1) << "\n";

  // TODO: Use varints
  const size_t header_size =
      4 + 4 + CountBits(coding.len_mask) + coding.num_syms + (K - 1) * (4);
  const size_t compressed_size = header_size + write_end[K - 1];
  std::string compressed;
  compressed.reserve(compressed_size);
  write_u32(compressed, raw.size());
  write_u32(compressed, coding.len_mask);
  for (int len = 0; len < 32; ++len) {
    int count = coding.len_count[len];
    if (count != 0) {
      compressed.push_back(count);
    }
  }
  compressed.append(reinterpret_cast<char*>(coding.sorted_syms),
                    coding.num_syms);
  for (int k = 0; k < K - 1; ++k) {
    write_u32(compressed, write_end[k]);
  }

  assert(compressed.size() == header_size);
  compressed.resize(compressed_size);
  const void* const read_base = raw.data();
  char* const write_base = compressed.data() + header_size;
  uint64_t write_index[K];
  for (int k = 0; k < K; ++k) {
    write_index[k] = write_end[k] - 8;
  }

  // Instead of using look up tables, AVX-512 registers and cross-lane shuffles
  // (vpermb) are used.
  vec64x8 hi_v[4];
  vec64x8 lo_v[4];
  {
    uint8_t hi_arr[256];
    uint8_t lo_arr[256];
    for (int c = 0; c < 256; ++c) {
      uint16_t code16 = coding.codes[c].bits << (16 - kMaxCodeLength);
      hi_arr[c] = code16 >> 8;
      // Pack length into the low byte instead of using separate arrays and
      // registers.
      lo_arr[c] = (code16 & 0xff) + coding.codes[c].len;
      assert((code16 & 15) == 0);
      assert(coding.codes[c].len <= 12);
    }
    for (int i = 0; i < 4; ++i) {
      hi_v[i] = _mm512_loadu_epi8(hi_arr + 64 * i);
      lo_v[i] = _mm512_loadu_epi8(lo_arr + 64 * i);
    }
  }

  // These variables are for storing the leftover state.
  uint64_t buf_arr[K];
  uint64_t buf_len_arr[K];
  const int M = K / 8;
  DEF_VECS(buf_v, _mm512_setzero_si512());
  DEF_VECS(buf_len_v, _mm512_setzero_si512());
  DEF_VECS(read_v, _mm512_loadu_epi64(read_index + 8 * m));
  DEF_VECS(write_v, _mm512_loadu_epi64(write_index + 8 * m));

  // Last stripe is always one of the smallest, so we can just base our loop
  // condition on that.
  for (size_t read_i = 0; read_i + 7 < sizes[K - 1]; read_i += 8) {
    const vec64x8 lohi_ctrl =
        _mm512_set4_epi64(0x0f0e'0d0c'0706'0504, 0x0b0a'0908'0302'0100,
                          0x0f0e'0d0c'0706'0504, 0x0b0a'0908'0302'0100);
    DEF_VECS(bytes, _mm512_i64gather_epi64(read_v[m], read_base, 1));
    FORM(m) {
      read_v[m] = _mm512_add_epi64(read_v[m], _mm512_set1_epi64(8));
      // Low and high bytes of the codes are combined using
      // _mm512_unpacklo_epi8 and _mm512_unpackhi_epi8 below. For these
      // instructions to work we must move the bytes corresponding to first and
      // second 4 symbols to lower and upper words of each 128-bit lane.
      //
      // By shuffling the input bytes we don't need to perform multiple
      // shuffles later.
      bytes[m] = _mm512_shuffle_epi8(bytes[m], lohi_ctrl);
    }

    // Decode all 64 bytes simultaneously.
    DEF_VECS(code_hi, _mm512_permutexvar_epi8(bytes[m], hi_v[0]));
    DEF_VECS(code_lo, _mm512_permutexvar_epi8(bytes[m], lo_v[0]));
    FORM(m) {
      UNROLL8 for (int k = 1; k < 4; ++k) {
        mask64 cmp =
            _mm512_cmpge_epu8_mask(bytes[m], _mm512_set1_epi8(char(k * 64)));
        code_hi[m] =
            _mm512_mask_permutexvar_epi8(code_hi[m], cmp, bytes[m], hi_v[k]);
        code_lo[m] =
            _mm512_mask_permutexvar_epi8(code_lo[m], cmp, bytes[m], lo_v[k]);
      }
    }

    // Now, we must rearrange and pack these codes. We'll do this by
    // processing 4 symbols for each stream at a time.
    UNROLL8 for (int j = 0; j < 2; ++j) {
      FORM(m) {
        vec64x8 pack16 = j == 0 ? _mm512_unpacklo_epi8(code_lo[m], code_hi[m])
                                : _mm512_unpackhi_epi8(code_lo[m], code_hi[m]);
        // Length was stored in low bits of `code_lo`
        vec64x8 len16 = _mm512_and_si512(pack16, _mm512_set1_epi16(0xf));
        // (Using "andnot" instead of "and" saves one register.)
        pack16 = _mm512_andnot_si512(_mm512_set1_epi16(0xf), pack16);
        // Now four codes are stored in the 16-bit words of each 64-bit integer
        // in `pack16`, with one 64-bit integer for each stream. `len16` stores
        // the length of each code in a 16-bit word at the same position.  Next
        // we must pack the codes by shifting.

        UNROLL8 for (int z = 0; z < 4; ++z) {
          const vec64x8 len = GetWord16(len16, z);
          const vec64x8 code_left = GetWord16ToTopBits(pack16, z);
          buf_v[m] = _mm512_or_epi64(
              buf_v[m], _mm512_srlv_epi64(code_left, buf_len_v[m]));
          buf_len_v[m] = _mm512_add_epi64(buf_len_v[m], len);
        }

        // Flush buffer
        vec8x64 num_bytes = _mm512_srli_epi64(buf_len_v[m], 3);
        _mm512_i64scatter_epi64(write_base, write_v[m], buf_v[m], 1);
        write_v[m] = _mm512_sub_epi64(write_v[m], num_bytes);
        // Shift these bytes out to the left from `buf_v`.
        vec8x64 written_bits =
            _mm512_andnot_epi64(_mm512_set1_epi64(7), buf_len_v[m]);
        buf_v[m] = _mm512_sllv_epi64(buf_v[m], written_bits);
        buf_len_v[m] = _mm512_and_epi64(buf_len_v[m], _mm512_set1_epi64(7));
      }
    }
  }
  FORM(m) {
    // Store vector registers back to C arrays.
    _mm512_storeu_epi64(buf_arr + 8 * m, buf_v[m]);
    _mm512_storeu_epi64(buf_len_arr + 8 * m, buf_len_v[m]);
    _mm512_storeu_epi64(read_index + 8 * m, read_v[m]);
    _mm512_storeu_epi64(write_index + 8 * m, write_v[m]);
  }

  // Write remaining bytes one stream at a time.
  for (int k = 0; k < K; ++k) {
    uint64_t buf = buf_arr[k];
    uint64_t buf_len = buf_len_arr[k];
    char* write_ptr = write_base + write_index[k];
    for (uint64_t i = read_index[k]; i < read_end[k]; ++i) {
      const uint8_t s = raw[i];
      auto code = coding.codes[s];
      buf |= uint64_t(code.bits) << (64 - buf_len - kMaxCodeLength);
      buf_len += code.len;
      // Flush:
      memcpy(write_ptr, &buf, 8);
      uint32_t num_bytes = buf_len >> 3;
      write_ptr -= num_bytes;
      buf <<= 8 * num_bytes;
      buf_len &= 7;
    }
  }

  return compressed;
}

template <int K, typename UsedDecoder>
std::string DecompressMultiAvx512Impl(std::string_view compressed) {
  static_assert(K % 8 == 0);
  constexpr int M = K / 8;
  const uint32_t raw_size = read_u32(compressed);
  const uint32_t len_mask = read_u32(compressed);
  uint8_t len_count[32] = {};
  int num_syms = 0;
  for (int i = 0; i < 32; ++i) {
    if (len_mask & (1 << i)) {
      len_count[i] = uint8_t(compressed[0]);
      compressed.remove_prefix(1);
      num_syms += len_count[i];
    }
  }

  UsedDecoder decoder(
      len_count, reinterpret_cast<const uint8_t*>(&compressed[0]), num_syms);
  compressed.remove_prefix(num_syms);

  uint64_t read_end_offset[K] = {};
  for (int k = 0; k < K - 1; ++k) {
    read_end_offset[k] = read_u32(compressed);
  }
  read_end_offset[K - 1] = compressed.size();

  int sizes[K] = {};
  for (int i = 0; i < K; ++i) {
    sizes[i] = raw_size / K;
  }
  for (size_t i = 0; i < raw_size % K; ++i) {
    ++sizes[i];
  }
  std::string raw(raw_size, 0);
  const uint8_t* const read_base =
      reinterpret_cast<const uint8_t*>(compressed.data());
  uint8_t* const write_base = reinterpret_cast<uint8_t*>(raw.data());
  const void* const dtable_base = decoder.dtable();

  uint64_t read_begin_offset[K] = {};
  for (int k = 1; k < K; ++k) {
    read_begin_offset[k] = read_end_offset[k - 1];
  }
  uint64_t read_offset[K] = {};
  for (int k = 0; k < K; ++k) {
    read_offset[k] = read_end_offset[k] - 8;
  }

  uint64_t write_offset[K] = {};
  for (int k = 1; k < K; ++k) {
    write_offset[k] = write_offset[k - 1] + sizes[k - 1];
  }

  uint64_t write_end[K] = {};
  for (int k = 0; k < K; ++k) {
    write_end[k] = write_offset[k] + sizes[k];
  }

  // 8 indices for reading data
  const vec8x64 zero_v = _mm512_setzero_si512();
  DEF_VECS(read_v, _mm512_loadu_epi64(read_offset + m * 8));
  DEF_VECS(write_v, _mm512_loadu_epi64(write_offset + m * 8));
  DEF_VECS(read_begin_v, _mm512_loadu_epi64(read_begin_offset + m * 8));
  DEF_VECS(write_limit_v,
           _mm512_sub_epi64(_mm512_loadu_epi64(write_end + m * 8),
                            _mm512_set1_epi64(7)));

  DEF_ARR(vec8x64, bits_consumed_v, zero_v);

  DEF_ARR(mask8, good, _mm512_cmplt_epi64_mask(write_v[m], write_limit_v[m]));

  auto some_good = [&good]() {
    mask8 all = good[0];
    UNROLL8 for (int m = 1; m < M; ++m) { all = _kor_mask8(all, good[m]); }
    return _cvtmask8_u32(all) != 0;
  };

  // Each iteration performs 4 decodes for each stream. With Decoder2x, each
  // decode can result in either 1 or 2 bytes decoded.
  while (some_good()) {
    DEF_VECS(write_len, zero_v);
    DEF_VECS(syms, zero_v);
    DEF_VECS(bits, zero_v);
    // Fill buffer.
    FORM(m) {
      // Update mask based on which streams still have output space left.
      good[m] = _kand_mask8(
          good[m], _mm512_cmplt_epi64_mask(write_v[m], write_limit_v[m]));

      const vec8x64 bytes_consumed = _mm512_srli_epi64(bits_consumed_v[m], 3);
      read_v[m] =
          _mm512_mask_sub_epi64(read_v[m], good[m], read_v[m], bytes_consumed);
      // Remainder bits: bits_consumed = bits_consumed % 8;
      bits_consumed_v[m] =
          _mm512_mask_and_epi64(bits_consumed_v[m], good[m], bits_consumed_v[m],
                                _mm512_set1_epi64(7));
      // Check that we can continue:
      good[m] = _kand_mask8(
          good[m], _mm512_cmpge_epi64_mask(read_v[m], read_begin_v[m]));

      // Read the bits to decompress
      bits[m] =
          _mm512_mask_i64gather_epi64(zero_v, good[m], read_v[m], read_base, 1);
      // Discard the used bits by shifting them out from the left.
      // code = (bits << bits_consumed) >> (64 - kMaxCodeLength).
      bits[m] = _mm512_sllv_epi64(bits[m], bits_consumed_v[m]);
    }
    UNROLL8 for (int j = 0; j < 4; ++j) {
      if constexpr (std::is_same_v<UsedDecoder, Decoder2x>) {
        // Table has 4 bytes for each entry. We read 64 bits for each entry
        // in order to keep the vectors 512 bits.

        // Here the index is calculated as `bits >> (64 - kMaxCodeLength)`.
        // Performing all of these gathers before doing any further processing
        // is a significant optimization on Zen5.
        DEF_VECS(dsyms, _mm512_i64gather_epi64(
                            _mm512_srli_epi64(bits[m], 64 - kMaxCodeLength),
                            dtable_base, 4));
        static_assert(sizeof(DecodedSym2x) == 4);
        // Now, code length is stored in the lowest byte of each 64-bit word,
        // decoded symbols in the next 2 bytes, and the number of syms in the
        // fourth byte.
        FORM(m) {
#if 1
          // The 64-byte vector contains symbols at byte positions
          // 1,2,9,10,17,18,...
          // Note: _mm512_shuffle_epi8 (vpshufb) shuffles across each 128-bit
          // lane, not across the whole 512-bit vector.
          const vec8x64 symbol_shuffle_ctrl =
              _mm512_set4_epi64(0xffffffffffff0a09, 0xffffffffffff0201,
                                0xffffffffffff0a09, 0xffffffffffff0201);
          vec8x64 these_syms =
              _mm512_shuffle_epi8(dsyms[m], symbol_shuffle_ctrl);
#else
          vec8x64 these_syms = _mm512_and_epi64(_mm512_srli_epi64(dsyms[m], 8),
                                                _mm512_set1_epi64(0xffff));
#endif

          // Store decoded symbols in the next bytes of `syms`.
          const vec8x64 sym_shift = _mm512_slli_epi64(write_len[m], 3);
          syms[m] = _mm512_or_epi64(syms[m],
                                    _mm512_sllv_epi64(these_syms, sym_shift));
          // Consume bits from input.
          const vec8x64 code_len =
              _mm512_and_epi64(dsyms[m], _mm512_set1_epi64(0xff));
          bits[m] = _mm512_sllv_epi64(bits[m], code_len);
          bits_consumed_v[m] = _mm512_mask_add_epi64(
              bits_consumed_v[m], good[m], bits_consumed_v[m], code_len);

          // The number of symbols (either 1 or 2) is stored in the fourth byte
          // of the decoder table entry.
#if 0
          vec8x64 num_decoded_syms = _mm512_and_epi64(
              _mm512_srli_epi64(dsyms[m], 24), _mm512_set1_epi64(0xff));
#else
          const vec8x64 num_decoded_syms_ctrl =
              _mm512_set4_epi64(0xffffffffffffff0b, 0xffffffffffffff03,
                                0xffffffffffffff0b, 0xffffffffffffff03);
          vec8x64 num_decoded_syms =
              _mm512_shuffle_epi8(dsyms[m], num_decoded_syms_ctrl);
#endif
          write_len[m] = _mm512_mask_add_epi64(write_len[m], good[m],
                                               write_len[m], num_decoded_syms);
        }
      } else {
        static_assert(std::is_same_v<UsedDecoder, Decoder1x>);
        // Table has two bytes for each entry. We read 64 bits for each entry
        // due to two reasons:
        // 1. There is no instruction to "gather" only 2 bytes
        // 2. Keeping vector size at 512 bits is faster since no conversions are
        // required.
        assert(decoder.max_symbols_decoded() == 1);
        DEF_VECS(dsyms, _mm512_i64gather_epi64(
                            _mm512_srli_epi64(bits[m], 64 - kMaxCodeLength),
                            dtable_base, 2));
        // Now, code length is stored in the lowest byte of each 64-bit word,
        // and the symbol in the second-lowest byte.
        FORM(m) {
          vec8x64 this_sym = _mm512_and_epi64(_mm512_srli_epi64(dsyms[m], 8),
                                              _mm512_set1_epi64(0xff));
          // Store decoded symbols in `syms`.
          syms[m] =
              _mm512_or_epi64(syms[m], _mm512_slli_epi64(this_sym, j * 8));
          __m512i code_len =
              _mm512_and_epi64(dsyms[m], _mm512_set1_epi64(0xff));
          // Consume bits
          bits[m] = _mm512_sllv_epi64(bits[m], code_len);
          bits_consumed_v[m] = _mm512_mask_add_epi64(
              bits_consumed_v[m], good[m], bits_consumed_v[m], code_len);
          write_len[m] = _mm512_mask_add_epi64(
              write_len[m], good[m], write_len[m], _mm512_set1_epi64(1));
        }
      }
    }

    FORM(m) {
      _mm512_mask_i64scatter_epi64(write_base, good[m], write_v[m], syms[m], 1);
      write_v[m] = _mm512_add_epi64(write_v[m], write_len[m]);
    }
  }
  // Read the rest using scalar code. This means we need to convert the
  // vectorized state to regular C++ variables.
  uint64_t bit_offset[K];
  for (int m = 0; m < M; ++m) {
    _mm512_storeu_epi64(write_offset + m * 8, write_v[m]);
    _mm512_storeu_epi64(read_offset + m * 8, read_v[m]);
    _mm512_storeu_epi64(bit_offset + m * 8, bits_consumed_v[m]);
  }

  // Decode 1 byte at a time:
  for (int k = 0; k < K; ++k) {
    DecodeSingleStream(decoder, read_base + read_begin_offset[k],
                       read_base + read_offset[k] + 8, bit_offset[k],
                       write_base + write_offset[k], write_base + write_end[k]);
  }
  return raw;
}

template <int K>
std::string DecompressMultiAvx512(std::string_view compressed) {
  return DecompressMultiAvx512Impl<K, Decoder2x>(compressed);
}

template std::string CompressMulti<2>(std::string_view raw);
template std::string DecompressMulti<2>(std::string_view compressed);
template std::string CompressMulti<3>(std::string_view raw);
template std::string DecompressMulti<3>(std::string_view compressed);
template std::string CompressMulti<4>(std::string_view raw);
template std::string DecompressMulti<4>(std::string_view compressed);

template std::string CompressMulti<8>(std::string_view raw);
template std::string DecompressMulti<8>(std::string_view compressed);

template std::string CompressMulti<32>(std::string_view raw);
template std::string DecompressMulti<32>(std::string_view compressed);

template std::string CompressMulti<16>(std::string_view raw);
template std::string DecompressMulti<16>(std::string_view compressed);

template std::string DecompressMultiAvx512<8>(std::string_view compressed);
template std::string DecompressMultiAvx512<16>(std::string_view compressed);
template std::string DecompressMultiAvx512<32>(std::string_view compressed);
template std::string CompressMultiAvx512<8>(std::string_view compressed);
template std::string CompressMultiAvx512<16>(std::string_view compressed);
template std::string CompressMultiAvx512<32>(std::string_view compressed);
}  // namespace huffman
