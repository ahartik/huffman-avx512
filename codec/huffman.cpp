#include "codec/huffman.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <bit>
#include <format>
#include <iostream>
#include <memory>
#include <vector>

#include "x86intrin.h"

namespace huffman {

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

struct BitCode {
  uint16_t bits;
  // uint16_t mask;
  int16_t len;
  // int16_t pad;
};

#ifdef HUFF_DEBUG

std::string SymToStr(uint8_t sym) {
  if (std::isprint(sym)) {
    return std::format("{:c}", sym);
  } else {
    return std::format("\\x{:02x}", sym);
  }
}

void PrintCode(char sym, uint64_t bits, int len) {
  std::cout << "Sym: " << SymToStr(sym) << ", code: " << bits << " ";
  if (len < 0 || len > 64) {
    std::cout << "BAD LEN: " << len << "\n";
    return;
  }
  for (int i = len - 1; i >= 0; --i) {
    std::cout << ((bits >> i) & 1);
  }
  std::cout << "\n" << std::flush;
  for (int i = len; i < 64; ++i) {
    assert(((bits >> i) & 1) == 0);
  }
}
#endif

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

// Maximum code length we want to use.
const int kMaxCodeLength = 16;
// Maximum code length that would be optimal in terms of compression.  We use
// shorter codes with slightly worse compression to gain better performance.
const int kMaxOptimalCodeLength = 32;

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
  // Tweak code lens to reduce max length.
  // This uses the "MiniZ" method as described in
  // https://create.stephan-brumme.com/length-limited-prefix-codes/#miniz

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
#ifdef HUFF_DEBUG
  std::cout << "LimitCodeLengths: adjustment_required " << adjustment_required
            << "\n";
  std::cout << "kraft_total: " << kraft_total << " one: " << one << "\n";
  std::cout << std::flush;
#endif
  int second_longest_len = kMaxCodeLength - 1;
  while (kraft_total > one) {
    // Decrease the length of one code with the maximum length.
    --len_count[kMaxCodeLength];
    // Increase the length for some code with currently a shorter length.
    while (second_longest_len > 0) {
      if (len_count[second_longest_len] > 0) {
        --len_count[second_longest_len];
        len_count[second_longest_len + 1] += 2;
        break;
      } else {
        --second_longest_len;
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
  uint64_t current_inc = 1ull << 16;
  for (int j = 0; j < coding.num_syms; ++j) {
    uint8_t sym = coding.sorted_syms[j];
    while (current_len_count == coding.len_count[current_len]) {
      ++current_len;
      current_inc >>= 1;
      current_len_count = 0;
    }
    coding.codes[sym].len = current_len;
    coding.codes[sym].bits = current_code >> (16 - current_len);
    // coding.codes[sym].mask = (1 << current_len) - 1;
#ifdef HUFF_DEBUG
    PrintCode(sym, coding.codes[sym].bits, current_len);
#endif
    current_code += current_inc;
    ++current_len_count;
  }
  // Code should wrap around perfectly once.
#ifdef HUFF_DEBUG
  std::cout << "compress: current_code at the end: " << current_code << "\n"
            << std::flush;
#endif
  assert(current_code == (1ull << 16));
  return coding;
}

class CodeWriter {
 public:
  explicit CodeWriter(char* output)
      : output_(output), cur_(0), free_bits_(64) {}
  CodeWriter() : output_(nullptr), cur_(0), free_bits_(64) {}

  void Init(char* output) {
    output_ = output;
    cur_ = 0;
    free_bits_ = 64;
  }

  void WriteCode(BitCode code) {
#if 1
    WriteLongCode(uint64_t(code.bits), code.len);
#else
    assert(free_bits_ >= 32);
    // XXX: Document this strange interface.
    cur_ |= uint64_t(code.bits) << (free_bits_ - 16);
    free_bits_ -= code.len;

    if (free_bits_ <= 32) {
      uint32_t to_write = htonl(cur_ >> 32);
      memcpy(output_, &to_write, 4);
      output_ += 4;
      free_bits_ += 32;
      cur_ <<= 32;
    }
#endif
  }

  void WriteLongCode(uint64_t bits, int num_bits) {
    // assert(free_bits_ > 0);
    assert(free_bits_ <= 64);
    if (free_bits_ <= num_bits) {
      // Write top bits and flush
      cur_ |= bits >> (num_bits - free_bits_);

      // std::cout << std::format("WROTE {:64b}\n", cur_);
      uint64_t to_write = __builtin_bswap64(cur_);
      memcpy(output_, &to_write, 8);
      output_ += 8;
      if (free_bits_ == num_bits) {
        // Avoid shift by 64
        cur_ = 0;
      } else {
        // Shift off bits we already wrote
        cur_ = bits << (64 - num_bits + free_bits_);
      }
      free_bits_ += 64 - num_bits;
    } else {
      cur_ |= bits << (free_bits_ - num_bits);
      free_bits_ -= num_bits;
    }
  }

  void Flush() {
    int num_bits = 64 - free_bits_;
    int num_bytes = num_bits >> 3;
    assert(num_bytes < 8);
    // std::cout << std::format("WROTE {:64b} num_bytes={}\n", cur_, num_bytes);
    uint64_t to_write = __builtin_bswap64(cur_);
    // XXX: Faster if we always write 8 bytes
    // memcpy(output_, &to_write, num_bytes);
    memcpy(output_, &to_write, 8);
    output_ += num_bytes;
    cur_ <<= 8 * num_bytes;
    free_bits_ += 8 * num_bytes;
  }

  void WriteFast(BitCode code) {
    assert(free_bits_ >= code.len);
    cur_ |= uint64_t(code.bits) << (free_bits_ - code.len);
    free_bits_ -= code.len;
  }

  void WriteFourCodes(BitCode a, BitCode b, BitCode c, BitCode d) {
#if 0
    uint64_t bits = uint64_t(a.bits);

    bits <<= b.len;
    bits |= uint64_t(b.bits);

    bits <<= c.len;
    bits |= uint64_t(c.bits);

    bits <<= d.len;
    bits |= uint64_t(d.bits);
    int len = a.len + b.len + c.len + d.len;
    WriteLongCode(bits, len);
#elif 1
    Flush();
    // Now we have at least 56 bits to use
    WriteFast(a);
    WriteFast(b);
    Flush();
    WriteFast(c);
    WriteFast(d);
#else
    uint64_t bits = (uint64_t(a.bits) << 48) | (uint64_t(b.bits) << 32) |
                    (uint64_t(c.bits) << 16) | (uint64_t(d.bits));
    uint64_t mask = (uint64_t(a.mask) << 48) | (uint64_t(b.mask) << 32) |
                    (uint64_t(c.mask) << 16) | (uint64_t(d.mask));
    bits = _pext_u64(bits, mask);
    int len = a.len + b.len + c.len + d.len;
    WriteLongCode(bits, len);
#endif
  }

  void WriteThreeCodes(BitCode a, BitCode b, BitCode c) {
    Flush();

    // This is slightly faster than repeated calls to WriteFast
    uint64_t bits = a.bits;
    bits <<= b.len;
    bits |= b.bits;
    bits <<= c.len;
    bits |= c.bits;

    int len = a.len + b.len + c.len;
    cur_ |= bits << (free_bits_ - len);
    free_bits_ -= len;
  }

  void Finish() {
    while (free_bits_ < 64) {
      uint8_t top = cur_ >> 56;
      *output_++ = top;
      cur_ <<= 8;
      free_bits_ += 8;
    }
  }

 private:
  char* output_;
  uint64_t cur_;
  int free_bits_;
};

class CodeReader {
 public:
  CodeReader() { Init(nullptr, 0); }

  CodeReader(const char* input, size_t size) { Init(input, size); }

  void Init(const char* input, size_t size) {
    input_ = input;
    end_ = input + size;
    buf_bits_ = 0;
    buf_len_ = 0;
    input_bits_used_ = 0;
    ConsumeBits(0);
  }

  uint32_t GetTopBits() const { return (buf_bits_ >> (32)) & 0xFfffFfff; }

  uint64_t GetTopBits64() const { return buf_bits_; }

  void ConsumeBits(int num_bits) {
    buf_len_ -= num_bits;

    uint64_t bytes = 0;
    if (__builtin_expect(input_ + 8 > end_, 0)) {
      int num_bytes_available = end_ - input_;
      if (num_bytes_available > 0) {
        memcpy(&bytes, input_, num_bytes_available);
      }
    } else {
      memcpy(&bytes, input_, 8);
    }

    bytes = FromBigEndian64(bytes);
    //   std::cout << std::format("Read {:016x} buf_len_={}\n", bytes, buf_len_)
    //             << std::flush;
    int num_bytes_to_read = (64 - buf_len_) >> 3;
    assert(buf_len_ >= 0);
    assert(num_bytes_to_read <= 8);

    buf_bits_ <<= num_bits;
    buf_bits_ |= bytes >> (buf_len_);
    buf_len_ += num_bytes_to_read * 8;
    assert(buf_len_ <= 64);
    assert(buf_len_ >= 48);
    //   std::cout << std::format("buf_bits_={:064B}\n", buf_bits_)
    //             << std::flush;
    input_ += num_bytes_to_read;
  }

 private:
  const char* input_;
  const char* end_;
  uint64_t buf_bits_;
  int input_bits_used_;
  int buf_len_;
};

struct DecodedSym {
  uint8_t sym;
  uint8_t code_len;
};

class Decoder {
 public:
  Decoder(uint8_t* len_count, const uint8_t* syms, int num_syms)
      : dtable_(1 << kMaxCodeLength) {
    // Note that this code also handles the strange case where there is only 1
    // symbol in the compressed text, in which case that symbol is encoded
    // using 0 bits.
    int current_len = 0;
    int current_len_count = 0;
    uint32_t current_code = 0;
    uint64_t current_inc = 1ull << 16;
    // std::cout << "num_syms: " << num_syms << "\n" << std::flush;
    for (int i = 0; i < num_syms; ++i) {
      while (len_count[current_len] == current_len_count) {
        ++current_len;
        current_len_count = 0;
        current_inc >>= 1;
        // Make sure there are no holes in this array.
        code_begin_[current_len] = current_code;
      }
      if (len_syms_[current_len] == nullptr) {
        len_syms_[current_len] = &syms[i];
        code_begin_[current_len] = current_code;
      }
      assert(current_len <= kMaxCodeLength);

      if (current_len <= 8) {
        uint32_t x_begin = current_code >> 8;
        for (uint32_t x = x_begin; x < x_begin + (current_inc >> 8); ++x) {
          assert(x < 256);
          len_begin_[x] = current_len;
          fast_sym_[x] = syms[i];
        }
      } else {
        uint32_t x = current_code >> 8;
        if (len_begin_[x] == 0) {
          len_begin_[x] = current_len;
        }
      }
      DecodedSym dsym = {syms[i], uint8_t(current_len)};
      std::fill(dtable_.begin() + current_code,
                dtable_.begin() + current_code + current_inc, dsym);

      current_code += current_inc;
      ++current_len_count;
    }
    // Should have exactly wrapped around:
    if (num_syms != 0) {
      assert(current_code == (1 << 16));
    }
  }

  int Decode(uint16_t code, uint8_t* out_sym) const {
#if 1
    DecodedSym dsym = dtable_[code];
    *out_sym = dsym.sym;
    return dsym.code_len;
#else
    // XXX: For some reason this old code doesn't work after modifying the codes
    // to be 16 bit.
    code >>= 16;
    const int top_byte = code >> 8;
    int len = len_begin_[top_byte];
#if 1
    // This may not be any faster.
    if (len <= 8) {
      // Fast path, know the symbol based on the top 8 bits alone.
      *out_sym = fast_sym_[top_byte];
      return len;
    }
#endif
    // We don't know the exact length of this code
    while ((code_begin_[len + 1] != 0) & (code_begin_[len + 1] <= code)) {
      ++len;
    }
#ifdef HUFF_DEBUG
    std::cout << "code len: " << len << "\n" << std::flush;
#endif
    int offset = (code - code_begin_[len]) >> (32 - len);
    uint8_t sym = len_syms_[len][offset];
#ifdef HUFF_DEBUG
    std::cout << std::format("Read {:032B} -> '{}'\n", code, SymToStr(sym));
    // std::cout << "code_len[sym] = " << int(code_len[sym]) << "\n" <<
    // std::flush; assert(len == code_len[sym]);
#endif
    *out_sym = sym;
    return len;
#endif
  }

 private:
  const uint8_t* len_syms_[32] = {};
  uint32_t code_begin_[32] = {};
  uint8_t len_begin_[256] = {};
  uint8_t fast_sym_[256] = {};
  std::vector<DecodedSym> dtable_;
};

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
  const int kSlop = 7;
  compressed.resize(header_size + (output_bits + 7) / 8 + kSlop);

  CodeWriter writer(&compressed[header_size]);
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
    writer.WriteThreeCodes(a, b, c);
  }
  while (input < end) {
    writer.WriteCode(coding.codes[*input++]);
  }
  writer.Finish();

  compressed.resize(compressed.size() - kSlop);
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
  Decoder decoder(
      len_count, reinterpret_cast<const uint8_t*>(compressed.data()), num_syms);
  compressed.remove_prefix(num_syms);

  std::string raw(raw_size, 0);
  CodeReader reader(compressed.data(), compressed.size());

  uint8_t* output = reinterpret_cast<uint8_t*>(raw.data());
  uint8_t* output_end = output + raw_size;
  // Three symbols at a time
  while (output + 2 < output_end) {
    uint64_t code = reader.GetTopBits64();
    int a_bits = decoder.Decode(code >> 48, output++);
    code <<= a_bits;
    int b_bits = decoder.Decode(code >> 48, output++);
    code <<= b_bits;
    int c_bits = decoder.Decode(code >> 48, output++);
    reader.ConsumeBits(a_bits + b_bits + c_bits);
  }
  // Last symbols
  while (output != output_end) {
    const uint64_t code = reader.GetTopBits64();
    int bits_read = decoder.Decode(code >> 48, output++);
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

  const int kSlop = 7;
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

  char* part_output[K];
  for (int k = 0; k < K; ++k) {
    part_output[k] =
        compressed.data() + header_size + ((k == 0) ? 0 : end_offset[k - 1]);
  }

  CodeWriter writer[K];
  for (int k = 0; k < K; ++k) {
    writer[k].Init(part_output[k]);
  }

  while (part_input[K - 1] + 2 < part_end[K - 1]) {
#pragma GCC unroll 8
    for (int k = 0; k < K; ++k) {
#if 0
      // We can write four codes of up to 16 bits per each flush.
      BitCode a = coding.codes[*part_input[k]++];
      BitCode b = coding.codes[*part_input[k]++];
      BitCode c = coding.codes[*part_input[k]++];
      BitCode d = coding.codes[*part_input[k]++];
      writer[k].WriteFourCodes(a, b, c, d);
#else
      // We can write three codes of up to 16 bits per each flush.
      BitCode a = coding.codes[*part_input[k]++];
      BitCode b = coding.codes[*part_input[k]++];
      BitCode c = coding.codes[*part_input[k]++];
      writer[k].Flush();
      writer[k].WriteFast(a);
      writer[k].WriteFast(b);
      writer[k].WriteFast(c);
#endif
    }
  }
  // Write potential last symbols.
  for (int k = 0; k < K; ++k) {
    while (part_input[k] != part_end[k]) {
      uint8_t sym = *part_input[k]++;
      BitCode code = coding.codes[sym];
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

  Decoder decoder(len_count, reinterpret_cast<const uint8_t*>(&compressed[0]),
                  num_syms);
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
#ifdef HUFF_DEBUG
  for (size_t i = 0; i < K; ++i) {
    std::cout << std::format("sizes[{}] = {}\n", i, sizes[i]);
  }
#endif
  CodeReader reader[K];
  for (int k = 0; k < K; ++k) {
    int start_index = (k == 0) ? 0 : end_offset[k - 1];
    reader[k].Init(compressed.data() + start_index,
                   end_offset[k] - start_index);
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

  while (part_output[K-1] + 2 < part_end[K-1]) {
#pragma GCC unroll 8
    for (int k = 0; k < K; ++k) {
      uint64_t code = reader[k].GetTopBits64();
      int a_len = decoder.Decode(code >> 48, part_output[k]++);
      code <<= a_len;
      int b_len = decoder.Decode(code >> 48, part_output[k]++);
      code <<= b_len;
      int c_len = decoder.Decode(code >> 48, part_output[k]++);
      reader[k].ConsumeBits(a_len + b_len + c_len);
    }
  }
  // Read last symbols.
  for (int k = 0; k < K; ++k) {
    while (part_output[k] != part_end[k]) {
      const uint32_t code = reader[k].GetTopBits();
      uint8_t sym;
      int bits_read = decoder.Decode(code >> 16, &sym);
      reader[k].ConsumeBits(bits_read);
      *part_output[k] = sym;
      ++part_output[k];
#ifdef HUFF_DEBUG
      std::cout << "Last sym " << k << " : " << SymToStr(sym) << "\n";
#endif
    }
  }

  return raw;
}

template std::string CompressMulti<2>(std::string_view compressed);
template std::string DecompressMulti<2>(std::string_view compressed);
template std::string CompressMulti<3>(std::string_view compressed);
template std::string DecompressMulti<3>(std::string_view compressed);
template std::string CompressMulti<4>(std::string_view compressed);
template std::string DecompressMulti<4>(std::string_view compressed);

}  // namespace huffman
