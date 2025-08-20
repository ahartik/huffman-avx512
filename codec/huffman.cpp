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

// Maximum code length we want to use.  Shorter max code lengths makes for
// faster compression and decompression.
const int kMaxCodeLength = 12;
const uint32_t kMaxCodeMask = (1 << kMaxCodeLength) - 1;
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
  // int second_longest_len = kMaxCodeLength - 1;
  while (kraft_total > one) {
    // Decrease the length of one code with the maximum length.
    --len_count[kMaxCodeLength];
    // Increase the length for some code with currently a shorter length.
    for (int j = kMaxCodeLength-1; j >= 0; --j) {
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
  uint64_t current_inc = 1ull << 16;
  for (int j = 0; j < coding.num_syms; ++j) {
    uint8_t sym = coding.sorted_syms[j];
    while (current_len_count == coding.len_count[current_len]) {
      ++current_len;
      current_inc >>= 1;
      current_len_count = 0;
    }
    coding.codes[sym].len = current_len;
    coding.codes[sym].bits = ReverseBits16(current_code);
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
  explicit CodeWriter(char* output) { Init(output); }

  CodeWriter() { Init(nullptr); }

  void Init(char* output) {
    output_ = output;
    buf_ = 0;
    buf_size_ = 0;
  }

  void WriteCode(BitCode code) {
    WriteFast(code);
    Flush();
  }

  void Flush() {
    int num_bytes = buf_size_ >> 3;
    assert(num_bytes <= 8);
    // This assumes little endian:
    memcpy(output_, &buf_, 8);
    output_ += num_bytes;
    buf_ >>= 8 * num_bytes;
    buf_size_ -= 8 * num_bytes;
  }

  void WriteFast(BitCode code) {
    assert(code.len + buf_size_ <= 64);
    buf_ |= uint64_t(code.bits) << buf_size_;
    buf_size_ += code.len;
  }

  void WriteThreeCodes(BitCode a, BitCode b, BitCode c) {
    WriteFast(a);
    WriteFast(b);
    WriteFast(c);
    Flush();
  }

  void Finish() {
    while (buf_size_ > 0) {
      uint8_t top = buf_ & 0xff;
      *output_++ = top;
      buf_ >>= 8;
      buf_size_ -= 8;
    }
  }

 private:
  char* output_;
  uint64_t buf_;
  int buf_size_;
};

class CodeReader {
 public:
  CodeReader() { Init(nullptr, 0); }

  CodeReader(const char* input, const char* end) { Init(input, end); }

  void Init(const char* input, const char* end) {
    input_ = input;
    end_ = end;
    buf_bits_ = 0;
    bits_used_ = 0;
    FillBuffer();
  }

  uint64_t GetFirstBits() const {
    return buf_bits_ >> bits_used_;
  }

  void ConsumeFast(int num_bits) {
    bits_used_ += num_bits;
  }

  void ConsumeBits(int num_bits) {
    ConsumeFast(num_bits);
    FillBuffer();
  }

  void FillBuffer() {
    input_ += (bits_used_ >> 3);
    bits_used_ &= 7;
    if (__builtin_expect(input_ + 8 > end_, 0)) {
      int num_bytes_available = end_ - input_;
      if (num_bytes_available > 0) {
        memcpy(&buf_bits_, input_, num_bytes_available);
      }
    } else {
      memcpy(&buf_bits_, input_, 8);
    }
  }

  bool FillBufferFast() {
    input_ += (bits_used_ >> 3);
    bits_used_ &= 7;
    if (__builtin_expect(input_ + 8 > end_, 0)) {
      return false;
    } else {
      memcpy(&buf_bits_, input_, 8);
    }
    return true;
  }

 private:
  const char* input_;
  const char* end_;
  uint64_t buf_bits_;
  int bits_used_;
};

struct DecodedSym {
  uint8_t code_len = 0;
  uint8_t sym = 0;
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
      }

      DecodedSym dsym = {
          .code_len = uint8_t(current_len),
          .sym = syms[i],
      };
      for (uint32_t code = ReverseBits16(current_code);
           code < (1 << kMaxCodeLength); code += (1 << current_len)) {
        dtable_[code] = dsym;
      }

      current_code += current_inc;
      ++current_len_count;
    }
    // Should have exactly wrapped around:
    if (num_syms != 0) {
      assert(current_code == (1 << 16));
    }
  }

  // TODO: Two symbols at a time decoding.

  int Decode(uint16_t code, uint8_t* out_sym) const {
    DecodedSym dsym = dtable_[code];
    const int len = dsym.code_len;
    *out_sym = dsym.sym;
    //     std::cout << std::format("Decode: {:016B} -> {}, {}\n", code,
    //     dsym.sym,
    //                              dsym.code_len);
    return len;
  }

 private:
  std::vector<DecodedSym> dtable_;
};

struct TrieNode {
  int max_len = 0;
  uint32_t mask = 0;
  DecodedSym* dsyms = nullptr;
};

class TrieDecoder {
 public:
  TrieDecoder(uint8_t* len_count, const uint8_t* syms, int num_syms)
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
      const uint8_t sym = syms[i];
      while (len_count[current_len] == current_len_count) {
        ++current_len;
        current_len_count = 0;
        current_inc >>= 1;
      }

      DecodedSym dsym = {
          .code_len = uint8_t(current_len),
          .sym = sym,
      };
      dsyms_[i] = dsym;
      // current_code is represented as first bit being the highest (position
      // 16), but we actually use the codes the other way around.
      const uint32_t le_code = ReverseBits16(current_code);
      int first_byte = le_code & 0xff;
      if (current_len <= 8) {
        for (uint32_t byte = first_byte; byte < 256;
             byte += (1 << current_len)) {
          trie_[byte].max_len = current_len;
          trie_[byte].dsyms = &dsyms_[i];
        }
      } else {
        // Possibly more than one length for this 8-bit prefix.
        // Lengths are added in increasing order.
        trie_[first_byte].max_len = current_len;
      }

      current_code += current_inc;
      ++current_len_count;
    }

    // Should have exactly wrapped around:
    if (num_syms != 0) {
      assert(current_code == (1 << 16));
    }
    int dtable_offset = 0;
    // Loop again to build the second level now that we have max_len.
    current_code = 0;
    current_inc = 1ull << 16;
    current_len = 0;
    current_len_count = 0;
    for (int i = 0; i < num_syms; ++i) {
      const uint8_t sym = syms[i];
      while (len_count[current_len] == current_len_count) {
        ++current_len;
        current_len_count = 0;
        current_inc >>= 1;
      }
      const uint32_t le_code = ReverseBits16(current_code);
      DecodedSym dsym = {
          .code_len = uint8_t(current_len),
          .sym = sym,
      };
      int byte = le_code & 0xff;
      if (trie_[byte].max_len > 8) {
        trie_[byte].mask = (1 << (trie_[byte].max_len - 8)) - 1;
        // Only these long codes need the second level table.
        if (trie_[byte].dsyms == nullptr) {
          trie_[byte].dsyms = &dtable_[dtable_offset];
          dtable_offset += 1 << (trie_[byte].max_len - 8);
          assert(size_t(dtable_offset) <= dtable_.size());
        }
        int num = 0;
        for (uint32_t c = le_code; c < (1u << trie_[byte].max_len);
             c += (1 << current_len)) {
          trie_[byte].dsyms[c >> 8] = dsym;
          ++num;
        }
        // std::cout << std::format("byte {:02x}, num={} max_len={}\n", byte,
        // num,
        //                          trie_[byte].max_len);
      }
      current_code += current_inc;
      ++current_len_count;
    }
  }

  int Decode(uint16_t code, uint8_t* out_sym) const {
    const TrieNode& node = trie_[code & 0xff];
    code >>= 8;
    code &= node.mask;
    DecodedSym dsym = node.dsyms[code];
    *out_sym = dsym.sym;
    return dsym.code_len;
  }

 private:
  std::vector<DecodedSym> dtable_;
  DecodedSym dsyms_[256];
  TrieNode trie_[256] = {};
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
    int a_bits = decoder.Decode(reader.GetFirstBits() & kMaxCodeMask, output++);
    reader.ConsumeFast(a_bits);
    int b_bits = decoder.Decode(reader.GetFirstBits() & kMaxCodeMask, output++);
    reader.ConsumeFast(b_bits);
    int c_bits = decoder.Decode(reader.GetFirstBits() & kMaxCodeMask, output++);
    reader.ConsumeFast(c_bits);
    int d_bits = decoder.Decode(reader.GetFirstBits() & kMaxCodeMask, output++);
    reader.ConsumeFast(d_bits);
    readers_good = reader.FillBufferFast();
    // reader.FillBuffer();
  }
  // Last symbols
  while (output != output_end) {
    reader.FillBuffer();
    const uint64_t code = reader.GetFirstBits();
    int bits_read = decoder.Decode(code & kMaxCodeMask, output++);
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
#ifdef HUFF_DEBUG
  for (size_t i = 0; i < K; ++i) {
    std::cout << std::format("sizes[{}] = {}\n", i, sizes[i]);
  }
#endif
  CodeReader reader[K];
  for (int k = 0; k < K; ++k) {
    int start_index = (k == 0) ? 0 : end_offset[k - 1];
    reader[k].Init(compressed.data() + start_index,
        compressed.data() + compressed.size());
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
        int code_len = decoder.Decode(reader[k].GetFirstBits() & kMaxCodeMask,
                                      part_output[k]++);
        reader[k].ConsumeFast(code_len);
      }
    }
#pragma GCC unroll 8
    for (int k = 0; k < K; ++k) {
      readers_good &= reader[k].FillBufferFast();
      // reader[k].FillBuffer();
    }
  }
#endif
  // Read last symbols.
  for (int k = 0; k < K; ++k) {
    reader[k].FillBuffer();
    while (part_output[k] != part_end[k]) {
      const uint64_t code = reader[k].GetFirstBits();
      int bits_read = decoder.Decode(code & kMaxCodeMask, part_output[k]++);
      reader[k].ConsumeBits(bits_read);
#ifdef HUFF_DEBUG
      std::cout << "Last sym " << k << " : " << SymToStr(*(part_output[k] - 1))
                << "\n";
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
