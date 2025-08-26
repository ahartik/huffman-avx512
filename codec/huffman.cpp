#include "codec/huffman.h"

#include <arpa/inet.h>
#include <immintrin.h>
#include <x86intrin.h>

#include <ctype.h>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <bit>
#include <format>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace huffman {

namespace {

// Maximum code length we want to use.  Shorter max code lengths makes for
// faster compression and decompression.
const int kMaxCodeLength = 13;
const uint32_t kMaxCodeMask = (1 << kMaxCodeLength) - 1;
// Maximum code length that would be optimal in terms of compression.  We use
// shorter codes with slightly worse compression to gain better performance.
const int kMaxOptimalCodeLength = 32;

#ifdef HUFF_DEBUG
#define HUFF_VLOG 1
#elif !defined(HUFF_VLOG)
#define HUFF_VLOG 0
#endif

#define DLOG(level) \
  if (level <= HUFF_VLOG) std::cout
}  // namespace

namespace internal {

void CountSymbols(std::string_view text, int* sym_count) {
  // Idea copied from Huff0: count in four stripes to maximize superscalar
  if (text.size() < 1500) {
    const size_t text_size = text.size();
    for (size_t i = 0; i < text_size; ++i) {
      ++sym_count[uint8_t(text[i])];
    }
  } else {
    // 4K, hopefully still fits on the stack.
    int tmp_count[4][256] = {};
    const size_t text_size = text.size();
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(text.data());
    const uint8_t* end = ptr + text_size;
    // TODO: This complexity is not worth it, make microbenchmarks and simplify
    // possibly.
    while (ptr + 7 < end) {
      uint64_t data;
      memcpy(&data, ptr, 8);
      ptr += 8;
      ++tmp_count[0][data & 0xff];
      ++tmp_count[1][(data >> 8) & 0xff];
      ++tmp_count[2][(data >> 16) & 0xff];
      ++tmp_count[3][(data >> 24) & 0xff];
      ++tmp_count[0][(data >> 32) & 0xff];
      ++tmp_count[1][(data >> 40) & 0xff];
      ++tmp_count[2][(data >> 48) & 0xff];
      ++tmp_count[3][(data >> 56) & 0xff];
    }
    while (ptr < end) {
      ++tmp_count[0][*ptr++];
    }
    for (int c = 0; c < 256; ++c) {
      sym_count[c] =
          tmp_count[0][c] + tmp_count[1][c] + tmp_count[2][c] + tmp_count[3][c];
    }
  }
}

}  // namespace internal
   //

using namespace huffman::internal;

namespace {

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

struct BitCode {
  uint16_t bits;
  // uint16_t mask;
  int16_t len;
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

struct CanonicalCoding {
  int sym_count[256] = {};
  BitCode codes[256] = {};
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

CanonicalCoding MakeCanonicalCoding(std::string_view text) {
  CanonicalCoding coding;
  if (text.empty()) {
    return coding;
  }
  CountSymbols(text, coding.sym_count);

  std::vector<Node> heap;
  std::vector<Node> tree;
  heap.reserve(256);
  for (int c = 0; c < 256; ++c) {
    if (coding.sym_count[c] != 0) {
      Node node;
      node.count = coding.sym_count[c];
      node.sym = uint8_t(c);
      heap.push_back(node);
      coding.sorted_syms[coding.num_syms] = node.sym;
      ++coding.num_syms;
    }
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
            [&](uint8_t a, uint8_t b) {
              return coding.sym_count[a] > coding.sym_count[b];
            });

  LimitCodeLengths(coding.len_count);

  for (int i = 0; i <= kMaxCodeLength; ++i) {
    if (coding.len_count[i] != 0) {
      coding.len_mask |= 1ull << i;
    }
  }

  int current_len = 0;
  int current_len_count = 0;
  uint32_t current_code = 0;
  uint64_t current_inc = 1 << kMaxCodeLength;
  for (int j = 0; j < coding.num_syms; ++j) {
    uint8_t sym = coding.sorted_syms[j];
    while (current_len_count == coding.len_count[current_len]) {
      ++current_len;
      current_inc >>= 1;
      current_len_count = 0;
    }
    coding.codes[sym].len = current_len;
    coding.codes[sym].bits = current_code;

    DLOG(1) << SymToStr(sym) << " -> " << coding.codes[sym] << "\n";

    current_code += current_inc;
    ++current_len_count;
  }
  // Code should wrap around perfectly once.
#ifdef HUFF_DEBUG
  std::cout << "compress: current_code at the end: " << current_code << "\n"
            << std::flush;
#endif
  assert(current_code == (1ull << kMaxCodeLength));
  return coding;
}

class CodeWriter {
 public:
  explicit CodeWriter(char* begin, char* end) { Init(begin, end); }

  CodeWriter() {
  }

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
    int num_bytes = buf_size_ >> 3;
    assert(num_bytes <= 8);
    // This assumes little endian:
    memcpy(output_, &buf_, 8);
    output_ -= num_bytes;
    // Leftmost bits were consumed, remaining move to the very left.
    buf_ <<= 8 * num_bytes;
    buf_size_ -= 8 * num_bytes;
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
    while (buf_size_ > 0) {
      uint8_t top = (buf_ >> 56) & 0xff;
      *out_byte-- = top;
      buf_ <<= 8;
      buf_size_ -= 8;
    }
  }

 private:
  char* begin_;
  char* end_;
  char* output_;
  uint64_t buf_;
  int buf_size_;
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

class Decoder {
 public:
  Decoder(uint8_t* len_count, const uint8_t* syms, int num_syms)
      : dtable_((1 << kMaxCodeLength) + 4) {
    // Note that this code also handles the strange case where there is only 1
    // symbol in the compressed text, in which case that symbol is encoded
    // using 0 bits.
    int current_len = 0;
    int current_len_count = 0;
    uint32_t current_code = 0;
    uint64_t current_inc = 1ull << kMaxCodeLength;
    // std::cout << "num_syms: " << num_syms << "\n" << std::flush;
    DLOG(1) << "Decoder:\n";
    for (int i = 0; i < num_syms; ++i) {
      while (len_count[current_len] == current_len_count) {
        max_code_for_len_[current_len] = current_code;
        ++current_len;
        current_len_count = 0;
        current_inc >>= 1;
      }

      DecodedSym dsym = {
          .code_len = uint8_t(current_len),
          .sym = syms[i],
      };
      DLOG(1) << SymToStr(syms[i]) << " -> "
              << BitCode(current_code, current_len) << "\n";
      std::fill(dtable_.begin() + current_code,
                dtable_.begin() + current_code + current_inc, dsym);

      current_code += current_inc;
      ++current_len_count;
    }
    for (int j = current_len; j <= 16; ++j) max_code_for_len_[j] = current_code;

    for (int j = 0; j < 16; ++j) {
      max_code_for_len_[j] -= 1;
    }
    // Should have exactly wrapped around:
    if (num_syms != 0) {
      assert(current_code == (1 << kMaxCodeLength));
    }
    max_code_vec_ =
        _mm256_loadu_si256(reinterpret_cast<__m256i*>(max_code_for_len_));
  }

  // TODO: Two symbols at a time decoding.

  inline int Decode(uint16_t code, uint8_t* out_sym,
                    bool try_avx = false) const {
    DLOG(2) << std::format("Decode({:016B}) \n", code);
    DecodedSym dsym = dtable_[code];

    *out_sym = dsym.sym;
    if (try_avx) {
      // This is slow and should likely be removed.
      // Kept here just in case we want to try this idea in AVX512 code in the
      // future.
      __m256i c_vec = _mm256_set1_epi16(code);
      // This limits code length to max 15 bits, since comparison is signed.
      __m256i gt_max = _mm256_cmpgt_epi16(c_vec, max_code_vec_);
      // Now, length is the same as the count of 0xffff words in `gt_max`.
      uint32_t gt_mask = _mm256_movemask_epi8(gt_max);
      const int len = (CountBits(gt_mask) / 2);
      if (HUFF_VLOG > 0 && len != dsym.code_len) {
        DLOG(2) << "AVX FAIL: len=" << len
                << " while dsym.code_len=" << int(dsym.code_len) << "\n";
        DLOG(2) << std::format("gt_mask = {:016B}\n", gt_mask) << std::flush;
        assert(false);
      }
      return len;
    } else {
      return dsym.code_len;
    }
  }

  const DecodedSym* dtable() const { return dtable_.data(); }

 private:
  std::vector<DecodedSym> dtable_;

  int16_t max_code_for_len_[16] = {};
  __m256i max_code_vec_;
};

using BestDecoder = Decoder;

}  // namespace

std::string Compress(std::string_view raw) {
  CanonicalCoding coding = MakeCanonicalCoding(raw);

  uint64_t output_bits = 0;
  for (int i = 0; i < coding.num_syms; ++i) {
    uint8_t sym = coding.sorted_syms[i];
    output_bits += coding.codes[sym].len * coding.sym_count[sym];
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
#pragma GCC unroll 1
  while (input + 2 < end) {
    // We can write three codes of up to 16 bits per each flush.
    BitCode a = coding.codes[*input++];
    BitCode b = coding.codes[*input++];
    BitCode c = coding.codes[*input++];
    // BitCode d = coding.codes[*input++];
    writer.WriteFast(a);
    writer.WriteFast(b);
    writer.WriteFast(c);
    writer.Flush();
  }
  while (input < end) {
    writer.WriteCode(coding.codes[*input++]);
  }
  writer.Finish();
  return compressed;
}

std::string Decompress(std::string_view compressed) {
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
  BestDecoder decoder(
      len_count, reinterpret_cast<const uint8_t*>(compressed.data()), num_syms);
  compressed.remove_prefix(num_syms);

  std::string raw(raw_size, 0);
  CodeReader reader(compressed.data(), compressed.data() + compressed.size());

  uint8_t* output = reinterpret_cast<uint8_t*>(raw.data());
  uint8_t* output_end = output + raw_size;
  // Four symbols at a time
  bool readers_good = true;
  while (readers_good & (output + 3 < output_end)) {
    int a_bits = decoder.Decode(reader.code(), output++);
    reader.ConsumeFast(a_bits);
    int b_bits = decoder.Decode(reader.code(), output++);
    reader.ConsumeFast(b_bits);
    int c_bits = decoder.Decode(reader.code(), output++);
    reader.ConsumeFast(c_bits);
    int d_bits = decoder.Decode(reader.code(), output++);
    reader.ConsumeFast(d_bits);
    readers_good = reader.FillBufferFast();
    // reader.FillBuffer();
  }
  // Last symbols
  while (output != output_end) {
    reader.FillBuffer();
    int bits_read = decoder.Decode(reader.code(), output++);
    reader.ConsumeBits(bits_read);
  }
  return raw;
}

template <int K>
std::string CompressMulti(std::string_view raw) {
  CanonicalCoding coding = MakeCanonicalCoding(raw);

  int sizes[K] = {};
  for (int i = 0; i < K; ++i) {
    sizes[i] = raw.size() / K;
  }
  for (size_t i = 0; i < raw.size() % K; ++i) {
    ++sizes[i];
  }
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

  const int kSlop = 8;
  // Compute starting positions for each part in the output.
  int end_offset[K] = {};
  {
    int pos = 0;
    for (int part = 0; part < K; ++part) {
      int64_t num_bits = 0;
      for (int i = 0; i < sizes[part]; ++i) {
        num_bits += coding.codes[uint8_t(raw[pos + i])].len;
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
      // std::cout << len << " ";
    }
  }
  // std::cout << "\n";
  compressed.append(reinterpret_cast<char*>(coding.sorted_syms),
                    coding.num_syms);
  for (int k = 0; k < K - 1; ++k) {
    write_u32(compressed, end_offset[k]);
  }

  // std::cout << compressed.size() << " == " << header_size << "\n" <<
  // std::flush;
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

#if 1
  while (part_input[K - 1] + 2 < part_end[K - 1]) {
#pragma GCC unroll 8
    for (int k = 0; k < K; ++k) {
      // We can write three codes of up to 16 bits per each flush.
      BitCode a = coding.codes[*part_input[k]++];
      BitCode b = coding.codes[*part_input[k]++];
      BitCode c = coding.codes[*part_input[k]++];
      writer[k].Flush();
      writer[k].WriteFast(a);
      writer[k].WriteFast(b);
      writer[k].WriteFast(c);
    }
  }
#endif
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

template <int K>
std::string DecompressMulti(std::string_view compressed) {
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

  BestDecoder decoder(
      len_count, reinterpret_cast<const uint8_t*>(&compressed[0]), num_syms);
  compressed.remove_prefix(num_syms);

  int end_offset[K] = {};
  for (int k = 0; k < K - 1; ++k) {
    end_offset[k] = read_u32(compressed);
  }
  end_offset[K - 1] = compressed.size();

  int sizes[K] = {};
  for (int i = 0; i < K; ++i) {
    sizes[i] = raw_size / K;
  }
  for (size_t i = 0; i < raw_size % K; ++i) {
    ++sizes[i];
  }
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

#if 1
  bool readers_good = true;
  while (readers_good & (part_output[K - 1] + 3 < part_end[K - 1])) {
#pragma GCC unroll 8
    for (int j = 0; j < 4; ++j) {
#pragma GCC unroll 8
      for (int k = 0; k < K; ++k) {
        int code_len = decoder.Decode(reader[k].code(), part_output[k]++,
                                      /* try_avx= */ false);
        reader[k].ConsumeFast(code_len);
      }
    }
#pragma GCC unroll 8
    for (int k = 0; k < K; ++k) {
      // The checks inside this call could be made faster for k > 0
      // if we were certain that decoding of k > 0 did not go further bcak than
      // for k == 0. This is certain for good inputs, but bad inputs
      // could be crafted where this is not the case. This optimization would
      // improve decompression speed by ~1%.
      readers_good &= reader[k].FillBufferFast();
      // assert(reader[k].is_fast());
    }
  }
#endif
  // Read last symbols.
  for (int k = 0; k < K; ++k) {
    reader[k].FillBuffer();
    while (part_output[k] != part_end[k]) {
      const uint64_t code = reader[k].code();
      int bits_read = decoder.Decode(code, part_output[k]++);
      reader[k].ConsumeBits(bits_read);
      DLOG(1) << "Last sym " << k << " : " << SymToStr(*(part_output[k] - 1))
              << "\n";
    }
  }

  return raw;
}

std::string Int64VecToString(__m512i vec) {
  int64_t nums[8];
  _mm512_storeu_epi64(nums, vec);
  std::ostringstream s;
  s << "[";
  for (int i = 0; i < 8; ++i) {
    s << nums[i];
    if (i != 7) {
      s << ", ";
    }
  }
  s << "]";
  return s.str();
}

// Slowish method for decoding a single stream. Used to finish off decoding one
// character at a time.
void DecodeSingleStream(const Decoder& decoder, const uint8_t* compressed_begin,
                        const uint8_t* compressed_end, int bit_offset,
                        uint8_t* out_begin, uint8_t* out_end) {
  CodeReader reader(reinterpret_cast<const char*>(compressed_begin),
                    reinterpret_cast<const char*>(compressed_end));
  reader.ConsumeBits(bit_offset);
  for (uint8_t* output = out_begin; output != out_end; ++output) {
    int bits = decoder.Decode(reader.code(), output);
    reader.ConsumeBits(bits);
  }
}

std::string DecompressMulti8Avx512(std::string_view compressed) {
  // TODO: this
  // constexpr int M = 1; // Superscalar parallelism
  constexpr int K = 8;  // SIMD parallelism
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

  Decoder decoder(len_count, reinterpret_cast<const uint8_t*>(&compressed[0]),
                  num_syms);
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
  __m512i read_index = _mm512_loadu_epi64(read_offset);
  __m512i write_index = _mm512_loadu_epi64(write_offset);

  const __m512i read_begin = _mm512_loadu_epi64(read_begin_offset);
  const __m512i write_limit =
      _mm512_sub_epi64(_mm512_loadu_epi64(write_end), _mm512_set1_epi64(7));

  // 8 integers for how many bits of the current word are already consumed.
  __m512i bits_consumed = _mm512_setzero_si512();
  const __m512i table_mask = _mm512_set1_epi64((1 << kMaxCodeLength) - 1);
  const __m512i zero_v = _mm512_setzero_si512();

  __mmask8 good = _cvtu32_mask8(0xff);

  // Each iteration decodes 4 bytes
  while (_cvtmask8_u32(good) != 0) {
    // Skip forward:
    // bytes_consumed = bits_consumed / 8;

    __m512i bytes_consumed = _mm512_srli_epi64(bits_consumed, 3);
    read_index =
        _mm512_mask_sub_epi64(read_index, good, read_index, bytes_consumed);
    // Remainder bits: bits_consumed = bits_consumed % 8;
    bits_consumed = _mm512_mask_and_epi64(bits_consumed, good, bits_consumed,
                                          _mm512_set1_epi64(7));

    // Check that we can continue:
    good = _kand_mask8(good, _mm512_cmplt_epi64_mask(write_index, write_limit));
    good = _kand_mask8(good, _mm512_cmpge_epi64_mask(read_index, read_begin));

    // Read the bits to decompress
    __m512i bits =
        _mm512_mask_i64gather_epi64(zero_v, good, read_index, read_base, 1);
    // Get code:
    // code = (bits << bits_consumed) >> (64 - kMaxCodeLength).
    bits = _mm512_sllv_epi64(bits, bits_consumed);

    __m512i syms = _mm512_setzero_si512();
#pragma GCC unroll 8
    for (int j = 0; j < 4; ++j) {
      __m512i index = _mm512_srli_epi64(bits, 64 - kMaxCodeLength);
      // Table has two bytes for each entry. We read 64 bits for each entry
      // due to two reasons:
      // 1. There is no instruction to "gather" only 2 bytes
      // 2. Keeping vector size at 512 bits is faster since no conversions are
      // required.
      __m512i dsyms = _mm512_i64gather_epi64(index, dtable_base, 2);
      // Now, code length is stored in the lowest byte of each 64-bit word, and
      // the symbol in the second-lowest byte.

      __m512i this_sym = _mm512_and_epi64(_mm512_srli_epi64(dsyms, 8),
                                          _mm512_set1_epi64(0xff));
      syms = _mm512_or_epi64(syms, _mm512_slli_epi64(this_sym, j * 8));
      __m512i code_len = _mm512_and_epi64(dsyms, _mm512_set1_epi64(0xff));
      // Consume bits
      bits = _mm512_sllv_epi64(bits, code_len);
      bits_consumed =
          _mm512_mask_add_epi64(bits_consumed, good, bits_consumed, code_len);
    }
    // Perform write:
    __m256i syms256 = _mm512_cvtepi64_epi32(syms);
    _mm512_mask_i64scatter_epi32(write_base, good, write_index, syms256, 1);
    write_index = _mm512_mask_add_epi64(write_index, good, write_index,
                                        _mm512_set1_epi64(4));
  }
  // Read the rest using scalar code. This means we need to convert the
  // vectorized state to more regular C++ variables.
  uint64_t bit_offset[K];
  _mm512_storeu_epi64(write_offset, write_index);
  _mm512_storeu_epi64(read_offset, read_index);
  _mm512_storeu_epi64(bit_offset, bits_consumed);

  // Decode 1 byte at a time:
  for (int k = 0; k < K; ++k) {
    DecodeSingleStream(decoder, read_base + read_begin_offset[k],
                       read_base + read_offset[k] + 8, bit_offset[k],
                       write_base + write_offset[k], write_base + write_end[k]);
  }
  return raw;
}

template std::string CompressMulti<2>(std::string_view compressed);
template std::string DecompressMulti<2>(std::string_view compressed);
template std::string CompressMulti<3>(std::string_view compressed);
template std::string DecompressMulti<3>(std::string_view compressed);
template std::string CompressMulti<4>(std::string_view compressed);
template std::string DecompressMulti<4>(std::string_view compressed);

template std::string CompressMulti<8>(std::string_view compressed);
template std::string DecompressMulti<8>(std::string_view compressed);

}  // namespace huffman
