#include "codec/huffman.h"

#include <ammintrin.h>
#include <arpa/inet.h>
#include <emmintrin.h>
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

#include "codec/histogram.h"

namespace huffman {

namespace {

// Maximum code length we want to use.  Shorter max code lengths makes for
// faster compression and decompression.
//
// Max code length 13 observes some cache misses with AVX-512 decompression,
// reducing performance slightly and increasing variance. Lower max code length
// is used to stay comparable to Huff0, which defaults to 11.
//
// NOTE: CompressMultiAvx512 relies on this length being <= 12.
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

[[maybe_unused]] void MyAbort() {
  std::cout << std::flush;
  std::cerr << std::flush;
  abort();
}

[[maybe_unused]] bool VecEqual(__m512i a, __m512i b) {
  __mmask8 k = _mm512_cmpeq_epi64_mask(a, b);
  uint32_t mask_i = _cvtmask8_u32(k);
  // std::cout << "mask_i = " << mask_i << "\n";
  return mask_i == 0xff;
}

[[maybe_unused]] std::string Int64VecToString(__m512i vec) {
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
using vec16x32 = __m512i;

using vec8x32 = __m256i;
using vec4x32 = __m128i;

using mask8 = __mmask8;
using mask16 = __mmask16;
using mask32 = __mmask32;
using mask64 = __mmask64;

int CountBits(uint64_t x) { return __builtin_popcountll(x); }

// Shuffle 16-bit words within each 64-bit element. Negative values lead to the
// corresponding word being zeroed.
//
// dst[0:15] = w0 < 0 ? 0 : src[w0*16:w0*16+15]
// and so on
vec64x8 ShuffleWords64(vec64x8 src, int w0, int w1, int w2, int w3) {
  uint64_t hi0 = w0 < 0 ? 0xf0 : (2 * w0 + 1);
  uint64_t lo0 = w0 < 0 ? 0xf0 : (2 * w0);
  uint64_t hi1 = w1 < 0 ? 0xf0 : (2 * w1 + 1);
  uint64_t lo1 = w1 < 0 ? 0xf0 : (2 * w1);
  uint64_t hi2 = w2 < 0 ? 0xf0 : (2 * w2 + 1);
  uint64_t lo2 = w2 < 0 ? 0xf0 : (2 * w2);
  uint64_t hi3 = w3 < 0 ? 0xf0 : (2 * w3 + 1);
  uint64_t lo3 = w3 < 0 ? 0xf0 : (2 * w3);
  uint64_t shuffle64 = (hi3 << 56) | (lo3 << 48) | (hi2 << 40) | (lo2 << 32) |
                       (hi1 << 24) | (lo1 << 16) | (hi0 << 8) | (lo0);
  //
  vec8x64 ctrl =
      _mm512_set4_epi64(shuffle64 + 0x0808'0808'0808'0808, shuffle64,
                        shuffle64 + 0x0808'0808'0808'0808, shuffle64);
  return _mm512_shuffle_epi8(src, ctrl);
}
vec8x64 GetWord16(vec8x64 vec, int W) {
  switch (W) {
    case 0:
      return _mm512_and_epi64(vec, _mm512_set1_epi64(0xffffULL));
    case 1: {
      return ShuffleWords64(vec, 1, -1, -1, -1);
    }
    case 2: {
      return ShuffleWords64(vec, 2, -1, -1, -1);
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
  // The prefix code is stored in the kMaxCodeLength least significant bits
  // such that the "first" bit is stored at the most significant (leftmost)
  // position. The bits past kMaxCodeLength are zero.
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
void ForallCodes(const uint16_t len_count[kMaxCodeLength + 1],
                 const uint8_t* syms, int num_syms, Func func) {
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
  uint16_t len_count[kMaxOptimalCodeLength + 1] = {};
  uint32_t len_mask = 0;
};

// Tweak code lens to reduce max length to kMaxCodeLength.
// This uses the "MiniZ" method as described in
// https://create.stephan-brumme.com/length-limited-prefix-codes/#miniz
void LimitCodeLengths(uint16_t* len_count) {
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
  assert(kraft_total == one);
}

void CollectCodeLen(int32_t children[256][2], int node, int len,
                    uint16_t* len_count) {
  if (node < 0) {
    ++len_count[len];
  } else {
    CollectCodeLen(children, children[node][0], len + 1, len_count);
    CollectCodeLen(children, children[node][1], len + 1, len_count);
  }
}

CanonicalCoding MakeCanonicalCoding(const ByteHistogram& sym_count) {
  CanonicalCoding coding;

  for (int c = 0; c < 256; ++c) {
    if (sym_count[c] != 0) {
      coding.sorted_syms[coding.num_syms] = uint8_t(c);
      ++coding.num_syms;
    }
  }
  if (coding.num_syms == 0) {
    return coding;
  }

  // Sort the symbols in decreasing order of frequency.
  std::sort(coding.sorted_syms, coding.sorted_syms + coding.num_syms,
            [&](uint8_t a, uint8_t b) { return sym_count[a] > sym_count[b]; });

  // Using the approach detailed at
  // https://en.wikipedia.org/wiki/Huffman_coding#Compression,
  // we maintain two queues: one for symbols (leaves) and one for internal tree
  // nodes.
  //
  // Our "heap" is implemented using these two queues, reading from
  // `coding.sorted_sym` and `tree_count`. The syms are sorted in descending
  // order of frequency, meaning we read from back to front.
  int next_sym = coding.num_syms - 1;
  uint32_t tree_count[256] = {};
  int next_tree_node = 0;
  int tree_size = 0;
  auto pop_min_node = [&]() -> std::pair<uint32_t, int> {
    bool pop_sym = false;
    if (next_sym >= 0) {
      if (next_tree_node == tree_size) {
        pop_sym = true;
      } else {
        int sym = coding.sorted_syms[next_sym];
        pop_sym = sym_count[sym] <= tree_count[next_tree_node];
      }
    }

    if (pop_sym) {
      assert(next_sym >= 0);
      uint32_t count = sym_count[coding.sorted_syms[next_sym]];
      --next_sym;
      // Leaves are just -1.
      // For the final codes we use canonical Huffman codes, the exact shape of
      // the tree does not matter.
      return {count, -1};
    } else {
      assert(next_tree_node < tree_size);
      int node = next_tree_node;
      uint32_t count = tree_count[node];
      ++next_tree_node;
      return {count, node};
    }
  };
  auto heap_size = [&]() -> int {
    return (tree_size - next_tree_node) + (next_sym + 1);
  };

  // To find the number of codes of each length, we store child links from
  // created internal nodes, and perform a recursive tree traversal at the end.
  // created nodes and
  //
  // This approach is different from Huff0, which collects parent links between
  // nodes. I found this current code to be just a hair bit faster than the
  // parent-link approach, and I thought presenting this alternative approach
  // might be more interesting.
  int32_t children[256][2];
  while (heap_size() > 1) {
    // Pop two elements and create a tree node.
    const auto a = pop_min_node();
    const auto b = pop_min_node();
    children[tree_size][0] = a.second;
    children[tree_size][1] = b.second;
    tree_count[tree_size] = a.first + b.first;
    ++tree_size;
  }
  const auto root = pop_min_node();
  CollectCodeLen(children, root.second, 0, coding.len_count);

  LimitCodeLengths(coding.len_count);

  for (int i = 0; i <= kMaxCodeLength; ++i) {
    if (coding.len_count[i] != 0) {
      coding.len_mask |= 1ull << i;
    }
  }

  // Build a "canonical Huffman code".
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
    assert(output_ >= begin_);
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
  Decoder1x(const uint16_t* len_count, const uint8_t* syms, int num_syms)
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
  // The order of bytes in memory is relied upon by the AVX512 code, so cannot
  // be changed without changing the AVX decoder as well.
  uint8_t num_bits_decoded;
  uint8_t syms[2];
  uint8_t num_syms;
};

class Decoder2x {
 public:
  Decoder2x(const uint16_t* len_count, const uint8_t* syms, int num_syms)
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
  auto hist = MakeHistogram(raw);
  CanonicalCoding coding = MakeCanonicalCoding(hist);

  uint64_t output_bits = 0;
  for (int i = 0; i < coding.num_syms; ++i) {
    uint8_t sym = coding.sorted_syms[i];
    output_bits += coding.codes[sym].len * hist[sym];
  }

  std::string compressed;
  // TODO: Fail for too long strings.
  write_u32(compressed, raw.size());
  if (raw.size() == 0) {
    // SPECIAL CASE FOR EMPTY STRING:
  }
  write_u32(compressed, coding.len_mask);
  for (uint16_t count : coding.len_count) {
    if (count != 0) {
      compressed.push_back(uint8_t(count));
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
  while (input != end) {
    writer.WriteCode(coding.codes[*input++]);
  }
  writer.Finish();
  return compressed;
}

struct ParsedHeader {
  uint32_t raw_size;
  uint16_t len_count[kMaxCodeLength + 1];
  const uint8_t* syms;
  int num_syms;
};
ParsedHeader ParseCompressedHeader(std::string_view& compressed) {
  // TODO: Validate header too, to prevent crashes on bad data.
  ParsedHeader header = {};
  header.raw_size = read_u32(compressed);
  const uint32_t len_mask = read_u32(compressed);

  const bool one_size = CountBits(len_mask) == 1;
  for (int i = 0; i <= kMaxCodeLength; ++i) {
    if (len_mask & (1 << i)) {
      header.len_count[i] = uint8_t(compressed[0]);
      if (one_size && compressed[0] == 0) {
        // std::cout << "ALL 8 BITS\n" << std::flush;
        assert(i == 8);
        header.len_count[i] = 256;
      }
      compressed.remove_prefix(1);
      header.num_syms += header.len_count[i];
    }
  }
  header.syms = reinterpret_cast<const uint8_t*>(compressed.data());
  compressed.remove_prefix(header.num_syms);
  return header;
}

template <typename UsedDecoder>
std::string DecompressImpl(std::string_view compressed) {
  const auto header = ParseCompressedHeader(compressed);
  UsedDecoder decoder(header.len_count, header.syms, header.num_syms);

  std::string raw(header.raw_size, 0);
  CodeReader reader(compressed.data(), compressed.data() + compressed.size());

  uint8_t* output = reinterpret_cast<uint8_t*>(raw.data());
  uint8_t* output_end = output + raw.size();
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

template <int T>
class CompressBase {
 public:
 private:
};

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
  std::vector<ByteHistogram> part_hist(K);
  ByteHistogram total_hist = {};
  {
    for (int k = 0; k < K; ++k) {
      part_hist[k] = MakeHistogram(std::string_view(
          reinterpret_cast<const char*>(part_input[k]), sizes[k]));
    }
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < 256; ++i) {
        total_hist[i] += part_hist[k][i];
      }
    }
  }
  CanonicalCoding coding = MakeCanonicalCoding(total_hist);

  const int kSlop = 8;
  // Compute starting positions for each part in the output.
  int end_offset[K] = {};
  {
    int pos = 0;
    for (int part = 0; part < K; ++part) {
      int64_t num_bits = 0;
      for (int c = 0; c < 256; ++c) {
        num_bits += part_hist[part][c] * coding.codes[c].len;
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

  // It's slightly strange, but ordering these loops like this is faster.
  // Other way around gets better instruction parallelism, but also has more
  // instructions so ends up slower.
  for (int k = 0; k < K; ++k) {
    CodeWriter writer(part_output[k], part_output[k + 1]);
    const uint8_t* const read_end = part_end[k];
    const uint8_t* read_ptr = part_input[k];
    while (read_ptr + 3 < read_end) {
      // We can write three codes of up to 14 bits per each flush.
      static_assert(kMaxCodeLength <= 14);
      UNROLL8 for (int j = 0; j < 4; ++j) {
        BitCode a = coding.codes[*read_ptr++];
        writer.WriteFast(a);
      }
      writer.Flush();
    }
    while (read_ptr != read_end) {
      BitCode code = coding.codes[*read_ptr++];
      writer.WriteCode(code);
    }
    writer.Finish();
  }

  return compressed;
}

template <int K, typename UsedDecoder>
std::string DecompressMultiImpl(std::string_view compressed) {
  const auto header = ParseCompressedHeader(compressed);
  UsedDecoder decoder(header.len_count, header.syms, header.num_syms);

  int end_offset[K] = {};
  for (int k = 0; k < K - 1; ++k) {
    end_offset[k] = read_u32(compressed);
  }
  end_offset[K - 1] = compressed.size();

  const auto sizes = SliceSizes<K>(header.raw_size);
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

  std::string raw(header.raw_size, 0);
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

// An AVX-512 ZMM* register holds 512 bits, which here is split into 8 64-bit
// integers. this means 8 streams are processed simultaneously. To achieve
// higher IPC, we also support multiples of 8 streams (16, 24, 32, ...).
//
// Symbols used in AVX-512 compression and decompression functions:
// - K = Number of streams
// - M = K / 8, number of 512-bit registers.
//
// The following macros are used to create and use arrays of SIMD variables.

// Defines an array of variables, one for every
#define DEF_ARR(type, name, val)                             \
  type name[M];                                              \
  do {                                                       \
    UNROLL8 for (int m = 0; m < M; ++m) { name[m] = (val); } \
  } while (0)

// The typical use case: define an array of 512-bit vectors.
#define DEF_VECS(name, val) DEF_ARR(vec8x64, name, val);

#define FORM(m) UNROLL8 for (int m = 0; m < M; ++m)

template <int K>
std::string CompressMultiAvx512Permute(std::string_view raw) {
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
  std::vector<ByteHistogram> part_hist(K);
  ByteHistogram total_hist = {};
  {
    for (int k = 0; k < K; ++k) {
      part_hist[k] = MakeHistogram(raw.substr(read_index[k], sizes[k]));
    }
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < 256; ++i) {
        total_hist[i] += part_hist[k][i];
      }
    }
  }
  CanonicalCoding coding = MakeCanonicalCoding(total_hist);

  const int kSlop = 8;
  // Compute starting positions for each part in the output.
  uint64_t write_end[K] = {};
  {
    int pos = 0;
    for (int part = 0; part < K; ++part) {
      int64_t num_bits = 0;
      for (int c = 0; c < 256; ++c) {
        num_bits += part_hist[part][c] * coding.codes[c].len;
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
        _mm512_set4_epi32(0x0f0e'0d0c,0x0706'0504, 0x0b0a'0908,0x0302'0100);
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

    // Look up codes for all 64 bytes simultaneously.
    // Each vpermb/_mm512_permutexvar_epi8 does a lookup of 64 elements.
    // The operation repeated 4 times and masked using comparisons to cover all
    // 256 possible byte values.
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
    // UNROLL8 doesn't help here.
    for (int j = 0; j < 2; ++j) {
      DEF_VECS(pack4,  j == 0 ? _mm512_unpacklo_epi8(code_lo[m], code_hi[m])
                                : _mm512_unpackhi_epi8(code_lo[m], code_hi[m]));
      // Length was stored in low bits of `code_lo`
      DEF_VECS(len4,  _mm512_and_si512(pack4[m], _mm512_set1_epi16(0xf)));
      // (Using "andnot" instead of "and" saves one register.)
      DEF_VECS(code4, _mm512_andnot_si512(_mm512_set1_epi16(0xf), pack4[m]));
      FORM(m) {
        // Now four codes are stored in the 16-bit words of each 64-bit integer
        // in `pack16`, with one 64-bit integer for each stream. `len16` stores
        // the length of each code in a 16-bit word at the same position.  Next
        // we must pack the codes by shifting.
        UNROLL8 for (int z = 0; z < 4; ++z) {
          const vec64x8 len = GetWord16(len4[m], z);
          const vec64x8 code_left = GetWord16ToTopBits(code4[m], z);
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

template <int K>
std::string CompressMultiAvx512Gather(std::string_view raw) {
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
  std::vector<ByteHistogram> part_hist(K);
  ByteHistogram total_hist = {};
  {
    for (int k = 0; k < K; ++k) {
      part_hist[k] = MakeHistogram(raw.substr(read_index[k], sizes[k]));
    }
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < 256; ++i) {
        total_hist[i] += part_hist[k][i];
      }
    }
  }
  CanonicalCoding coding = MakeCanonicalCoding(total_hist);

  const int kSlop = 8;
  // Compute starting positions for each part in the output.
  uint64_t write_end[K] = {};
  {
    int pos = 0;
    for (int part = 0; part < K; ++part) {
      int64_t num_bits = 0;
      for (int c = 0; c < 256; ++c) {
        num_bits += part_hist[part][c] * coding.codes[c].len;
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

  // Slightly modified version of BitCode.
  struct BitCodeForGather {
    uint16_t len;
    // Unlike BitCode::bits, this has the first bit be the 15th bit, i.e. the
    // most significant bit of a 16-bit word.
    uint16_t bits;
  };
  BitCodeForGather codes[256 + 1];
  for (int i = 0; i < 256; ++i) {
    codes[i].len = coding.codes[i].len;
    codes[i].bits = coding.codes[i].bits << (16 - kMaxCodeLength);
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
    DEF_VECS(bytes, _mm512_i64gather_epi64(read_v[m], read_base, 1));
    FORM(m) { read_v[m] = _mm512_add_epi64(read_v[m], _mm512_set1_epi64(8)); }

    UNROLL8 for (int half = 0; half < 2; ++half) {
      UNROLL8 for (int z = 0; z < 4; ++z) {
        FORM(m) {
          // XXX: This can be optimized slightly:
          const vec64x8 current_byte =
              _mm512_and_epi64(_mm512_srli_epi64(bytes[m], half * 32 + z * 8),
                               _mm512_set1_epi64(0xff));
          const vec64x8 bitcode = _mm512_i64gather_epi64(
              current_byte, codes, sizeof(BitCodeForGather));
          const vec64x8 code_len =
              _mm512_and_epi64(bitcode, _mm512_set1_epi64(0xff));
          const vec64x8 code_left = GetWord16ToTopBits(bitcode, 1);

          buf_v[m] = _mm512_or_epi64(
              buf_v[m], _mm512_srlv_epi64(code_left, buf_len_v[m]));
          buf_len_v[m] = _mm512_add_epi64(buf_len_v[m], code_len);
        }
      }
      FORM(m) {
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
    assert(write_index[k] < (1 << 30));
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

template <int K>
std::string CompressMultiAvx512(std::string_view raw) {
#ifdef HUFF_COMPRESS_GATHER
  return CompressMultiAvx512Gather<K>(raw);
#else
  return CompressMultiAvx512Permute<K>(raw);
#endif
}

template <int K, typename UsedDecoder>
std::string DecompressMultiAvx512Impl(std::string_view compressed) {
  static_assert(K % 8 == 0);
  constexpr int M = K / 8;
  const auto header = ParseCompressedHeader(compressed);
  UsedDecoder decoder(header.len_count, header.syms, header.num_syms);

  uint64_t read_end_offset[K] = {};
  for (int k = 0; k < K - 1; ++k) {
    read_end_offset[k] = read_u32(compressed);
  }
  read_end_offset[K - 1] = compressed.size();

  int sizes[K] = {};
  for (int i = 0; i < K; ++i) {
    sizes[i] = header.raw_size / K;
  }
  for (size_t i = 0; i < header.raw_size % K; ++i) {
    ++sizes[i];
  }
  std::string raw(header.raw_size, 0);
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

  // Decode 1-2 bytes at a time:
  for (int k = 0; k < K; ++k) {
    DecodeSingleStream(decoder, read_base + read_begin_offset[k],
                       read_base + read_offset[k] + 8, bit_offset[k],
                       write_base + write_offset[k], write_base + write_end[k]);
  }
  return raw;
}

template <int K>
std::string DecompressMultiAvx512Gather(std::string_view compressed) {
  return DecompressMultiAvx512Impl<K, Decoder2x>(compressed);
}

template <int K>
std::string DecompressMultiAvx512Permute(std::string_view compressed) {
  static_assert(K % 8 == 0);
  constexpr int M = K / 8;
  const auto header = ParseCompressedHeader(compressed);

  uint64_t read_end_offset[K] = {};
  for (int k = 0; k < K - 1; ++k) {
    read_end_offset[k] = read_u32(compressed);
  }
  read_end_offset[K - 1] = compressed.size();

  int sizes[K] = {};
  for (int i = 0; i < K; ++i) {
    sizes[i] = header.raw_size / K;
  }
  for (size_t i = 0; i < header.raw_size % K; ++i) {
    ++sizes[i];
  }
  DLOG(1) << __func__ << ": raw_size = " << header.raw_size << "\n";
  std::string raw(header.raw_size, 0);
  const uint8_t* const read_base =
      reinterpret_cast<const uint8_t*>(compressed.data());
  uint8_t* const write_base = reinterpret_cast<uint8_t*>(raw.data());

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

  // Decoding method:
  // 1. For each of the 8 streams (64-bit element), repeat the current
  //    code (topmost bits) 4 times
  // 2. Perform comparisons against starting codes for each code length,
  //    to determine the code length for each stream.
  // 3. Produce a vector of starting codes corresponding to these code
  //    lengths, suitably adjusted such that index to sorted symbols can
  //    be found by subtraction.
  //  4. Once 8 such indices have been decoded for each stream (total 64
  //     indices = 64 bytes), look up the symbols ("raw" bytes) using
  //     `_mm512_permutexvar_epi8`

  // Prepare data for lookups: build vectors for looking up code length by
  // comparisons.
  uint16_t first_code_for_len[17] = {};
  uint16_t code_len_offset[17] = {};
  int last_len = -1;
  {
    int sym_i = 0;
    ForallCodes(header.len_count, header.syms, header.num_syms,
                [&](uint8_t sym, BitCode code) {
                  assert(sym == header.syms[sym_i]);
                  while (int(code.len) > last_len) {
                    ++last_len;
                    first_code_for_len[last_len] = code.bits;
                    // This offset is used to find the corresponding symbol
                    // index from a code when the length of the code is known.
                    uint16_t short_code =
                        code.bits >> (kMaxCodeLength - code.len);
                    assert(short_code >= sym_i);
                    code_len_offset[last_len] = short_code - sym_i;
                  }
                  ++sym_i;
                  return true;
                });
    assert(sym_i == header.num_syms);
    for (int i = last_len + 1; i <= 16; ++i) {
      first_code_for_len[i] = 0xffff;
    }
  }
  DLOG(1) << "last_len = " << last_len << "\n";
  // We rely on kMaxCodeLength <= 12, which means we only need three
  static_assert(kMaxCodeLength <= 12);
  vec8x64 first_code_v[3];
  vec8x64 decode_data_v[3];
  for (int i = 0; i < 3; ++i) {
    uint64_t packed_first_code = 0;
    uint64_t packed_data = 0;
    for (int j = 0; j < 4; ++j) {
      // We don't support zero-length code here.
      int len = i * 4 + j + 1;
      if (len <= last_len) {
        // Here the code is shifted so that the first bit is the most
        // significant bit of a 16-bit word.
        packed_first_code |= uint64_t(first_code_for_len[len])
                             << (j * 16 + 16 - kMaxCodeLength);
        // Length is stored at the top 4 bits of the "decode data". This
        // allows the length to be extracted after lookup. Also, the codes will
        // be ordered by length.
        packed_data |= (uint64_t(len << 12) | code_len_offset[len]) << (j * 16);
      } else {
        packed_first_code |= 0xffffULL << (j * 16);
        // If we encounter FFFF in the stream we still need to decode that as
        // `last_len`.
        packed_data |= (uint64_t(last_len << 12) | code_len_offset[last_len])
                       << (j * 16);
      }
    }
    DLOG(1) << std::format("packed_first_code[{}] = {:016x}\n", i,
                           packed_first_code);
    DLOG(1) << std::format("packed_data[{}]       = {:016x}\n", i, packed_data);
    first_code_v[i] = _mm512_set1_epi64(packed_first_code);
    decode_data_v[i] = _mm512_set1_epi64(packed_data);
  }

  // Prepare permute vectors for looking up symbols.
  vec8x64 sym_lookup_v[4];
  {
    uint8_t sym_arr[256];
    uint8_t* sym_end =
        std::copy(header.syms, header.syms + header.num_syms, sym_arr);
    std::fill(sym_end, sym_arr + 256, 0xfe);
    for (int i = 0; i < 4; ++i) {
      sym_lookup_v[i] = _mm512_loadu_epi8(sym_arr + 64 * i);
    }
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
    FORM(m) {
      // Update mask based on which streams still have output space left.
      good[m] = _kand_mask8(
          good[m], _mm512_cmplt_epi64_mask(write_v[m], write_limit_v[m]));
      // Check that we can continue:
      good[m] = _kand_mask8(
          good[m], _mm512_cmpge_epi64_mask(read_v[m], read_begin_v[m]));
    }
    // For each iteration we decode 8 symbols per stream.
    //
    // Conversion from symbol indices (into header.syms) are translated later
    // into decoded symbols.
    DEF_VECS(sym_i, zero_v);
    // Read "goodness" is tracked separately inside each iteration, since we
    // may run out of data to read after producing 4 bytes. In that case we
    // still want to do the write, but reading will stop after the first half,
    // and overall `good` mask will be updated for the next iteration.
    DEF_ARR(mask8, read_good, good[m]);
    DEF_VECS(write_len, zero_v);
    // 8 symbols can take more than 8 bytes in compressed form sometimes,
    // so we perform two reads of 64 bits, each producing 4 decoded symbols.

    // Unrolling is a small optimization here:
    UNROLL8 for (int half = 0; half < 2; ++half) {
      DEF_VECS(bits, zero_v);
      FORM(m) {
        // Read the bits to decompress

        const vec8x64 bytes_consumed = _mm512_srli_epi64(bits_consumed_v[m], 3);
        read_v[m] = _mm512_mask_sub_epi64(read_v[m], read_good[m], read_v[m],
                                          bytes_consumed);
        // Remainder bits: bits_consumed = bits_consumed % 8;
        // pauli was here
        bits_consumed_v[m] =
            _mm512_mask_and_epi64(bits_consumed_v[m], read_good[m],
                                  bits_consumed_v[m], _mm512_set1_epi64(7));
        // Note: because we check for read "goodness" here, it's possible for
        // one iteration of the overall loop to produce 0,4, or 8 bytes for
        // each stream.
        read_good[m] = _kand_mask8(
            read_good[m], _mm512_cmpge_epi64_mask(read_v[m], read_begin_v[m]));
        write_len[m] = _mm512_mask_add_epi64(
            write_len[m], read_good[m], write_len[m], _mm512_set1_epi64(4));
        bits[m] = _mm512_mask_i64gather_epi64(zero_v, read_good[m], read_v[m],
                                              read_base, 1);
        // Discard previously used bits by shifting them out from the left.
        // code = (bits << bits_consumed) >> (64 - kMaxCodeLength).
        bits[m] = _mm512_sllv_epi64(bits[m], bits_consumed_v[m]);
      }
      //
      DEF_VECS(code4, zero_v);
      DEF_VECS(decode_data4, zero_v);

      // Unrolling this loop actually makes the code slower
      // UNROLL8
      for (int byte_i = 0; byte_i < 4; ++byte_i) {
        // Repeat the next code (most significant bits) four times in each
        // 64-bit element:
        DEF_VECS(code_repeat, ShuffleWords64(bits[m], 3, 3, 3, 3));
        DEF_VECS(decode_data, zero_v);

        UNROLL8 for (int len_i = 0; len_i < 3; ++len_i) {
          FORM(m) {
            mask32 ge =
                _mm512_cmpge_epu16_mask(code_repeat[m], first_code_v[len_i]);
            decode_data[m] = _mm512_mask_blend_epi16(ge, decode_data[m],
                                                     decode_data_v[len_i]);
          }
        }

        // Now each 16-bit word contains some potentially-valid decode data.
        // The correct one is the largest one.
        FORM(m) {
          decode_data[m] = _mm512_max_epu16(
              decode_data[m], _mm512_rol_epi64(decode_data[m], 16));
          decode_data[m] = _mm512_max_epu16(
              decode_data[m], _mm512_rol_epi64(decode_data[m], 32));
          // Since we used rotates, the correct decode data is now found in
          // each 16-bit word. We store the code and the found data in the
          // corresponding spots in `code4` and `decode_data4`.
          const vec8x64 word_mask =
              _mm512_set1_epi64(0xffffull << (byte_i * 16));
          code4[m] = _mm512_or_epi64(
              code4[m], _mm512_and_epi64(code_repeat[m], word_mask));
          decode_data4[m] = _mm512_or_epi64(
              decode_data4[m], _mm512_and_epi64(decode_data[m], word_mask));

          vec8x64 code_len = _mm512_srli_epi64(decode_data[m], 60);
          bits_consumed_v[m] = _mm512_mask_add_epi64(
              bits_consumed_v[m], read_good[m], bits_consumed_v[m], code_len);
          bits[m] = _mm512_sllv_epi64(bits[m], code_len);
          DLOG(2) << std::format("bits[{}] = {}, byte_i={}\n", m,
                                 Int64VecToString(bits[m]), byte_i)
                  << std::format("code_repeat[{}] = {}, byte_i={}\n", m,
                                 Int64VecToString(code_repeat[m]), byte_i)
                  << std::format("code_len[{}] = {}, byte_i={}\n", m,
                                 Int64VecToString(code_len), byte_i);
        }
      }
      // Now, decode 4 symbols for each stream using the data we gathered:
      FORM(m) {
        // These variables contain data in all 16-bit elements.
        const vec8x64 decode_offset =
            _mm512_and_epi64(decode_data4[m], _mm512_set1_epi16(0xfff));
        const vec8x64 code_len4 = _mm512_srli_epi16(decode_data4[m], 12);
        // Symbol index is found by:
        // sym_index = (code >> (16 - code_len)) - decode_offset
        const vec8x64 sym_index = _mm512_sub_epi16(
            _mm512_srlv_epi16(
                code4[m], _mm512_sub_epi16(_mm512_set1_epi16(16), code_len4)),
            decode_offset);
        // Now `sym_index` holds the indices for these four symbols as 16-bit
        // integer values. Convert these to 8-bit values and order them so that
        // the first four bytes correspond to the the first four bytes of
        // `sym_i` and so on.
        if (half == 0) {
          // Do the processing after the second half.
          sym_i[m] = sym_index;
        } else {
          assert(half == 1);
          //
          sym_i[m] = _mm512_or_si512(sym_i[m], _mm512_slli_epi64(sym_index, 8));
          // Sym index contains 8-bit indices each stored as a 16-bit word. Next
          // move them to their proper spot in `sym_i`.
          const vec8x64 sym_shuffle_ctrl = _mm512_set4_epi32(
              0x0f0d'0b09, 0x0e0c'0a08, 0x0705'0301, 0x0604'0200  //
          );
          sym_i[m] = _mm512_shuffle_epi8(sym_i[m], sym_shuffle_ctrl);
        }
        DLOG(2) << "COMBINED \n";
        DLOG(2) << std::format("code4[{}] = {} \n", m,
                               Int64VecToString(code4[m]))
                << std::format("decode_data4[{}] = {}\n", m,
                               Int64VecToString(decode_data4[m]))
                << std::format("code_len4[{}] = {}\n", m,
                               Int64VecToString(code_len4));
        DLOG(2) << "sym_i now: " << Int64VecToString(sym_i[m]) << "\n";
      }
    }
    // Symbol indices have been decoded, now they need to be converted to
    // symbols. This is done by repeated masked vpermb instructions.
    DEF_VECS(syms, _mm512_permutexvar_epi8(sym_i[m], sym_lookup_v[0]));
    for (int j = 1; j < 4; ++j) {
      FORM(m) {
        mask64 ge =
            _mm512_cmpge_epu8_mask(sym_i[m], _mm512_set1_epi8(char(64 * j)));
        syms[m] = _mm512_mask_permutexvar_epi8(syms[m], ge, sym_i[m],
                                               sym_lookup_v[j]);
      }
    }

    FORM(m) {
      _mm512_mask_i64scatter_epi64(write_base, good[m], write_v[m], syms[m], 1);
      write_v[m] = _mm512_add_epi64(write_v[m], write_len[m]);
      good[m] = _kand_mask8(good[m], read_good[m]);
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

  // "regular" decoder is used for the leftover symbols.
  Decoder1x decoder(header.len_count, header.syms, header.num_syms);
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
#ifdef HUFF_DECOMPRESS_PERMUTE
  return DecompressMultiAvx512Permute<K>(compressed);
#else
  return DecompressMultiAvx512Gather<K>(compressed);
#endif
}

#define INSTANTIATE_SCALAR(K)                                  \
  template std::string CompressMulti<K>(std::string_view raw); \
  template std::string DecompressMulti<K>(std::string_view compressed)

#define INSTANTIATE_AVX(K)                                                    \
  template std::string CompressMultiAvx512<K>(std::string_view raw);          \
  template std::string DecompressMultiAvx512<K>(std::string_view compressed); \
  template std::string CompressMultiAvx512Permute<K>(                       \
      std::string_view compressed);                                           \
  template std::string DecompressMultiAvx512Permute<K>(                     \
      std::string_view compressed);                                           \
  template std::string DecompressMultiAvx512Gather<K>(                        \
      std::string_view compressed);                                           \
  template std::string CompressMultiAvx512Gather<K>(std::string_view compressed)

INSTANTIATE_SCALAR(1);
INSTANTIATE_SCALAR(2);
INSTANTIATE_SCALAR(4);
INSTANTIATE_SCALAR(8);
INSTANTIATE_SCALAR(16);
INSTANTIATE_SCALAR(32);

INSTANTIATE_AVX(8);
INSTANTIATE_AVX(16);
INSTANTIATE_AVX(24);
INSTANTIATE_AVX(32);
INSTANTIATE_AVX(40);
INSTANTIATE_AVX(48);

}  // namespace huffman
