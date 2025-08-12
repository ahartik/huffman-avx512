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

namespace huffman {
namespace {

struct BitCode {
  uint32_t bits;
  int len;
};

#ifdef HUFF_DEBUG

std::string SymToStr(uint8_t sym) {
  if (std::isprint(sym)) {
    return std::format("{:c}", sym);
  } else {
    return std::format("\\x{:02x}", sym);
  }
}

void PrintCode(char sym, uint32_t bits, int len) {
  std::cout << "Sym: " << SymToStr(sym) << ", code: ";
  if (len < 0 || len > 32) {
    std::cout << "BAD LEN: " << len << "\n";
    return;
  }
  for (int i = 0; i < len; ++i) {
    std::cout << ((bits >> (31 - i)) & 1);
  }
  std::cout << "\n";
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
                  uint8_t* code_len) {
  if (node->children[0] == node->children[1]) {
    code_len[node->sym] = len;
  } else {
    get_code_len(tree, &tree[node->children[0]], len + 1, code_len);
    get_code_len(tree, &tree[node->children[1]], len + 1, code_len);
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
  uint8_t code_len[256] = {};
  BitCode codes[256] = {};
  uint8_t sorted_syms[256] = {};
  int num_syms = 0;
  uint8_t len_count[32] = {};
  uint32_t len_mask = 0;
};

CanonicalCoding MakeCanonicalCoding(std::string_view text) {
  CanonicalCoding coding;
  if (text.empty()) {
    return coding;
  }
  for (const unsigned char c : text) {
    ++coding.sym_count[c];
  }
  std::vector<Node> heap;
  std::vector<Node> tree;
  std::vector<uint8_t> syms;
  for (int c = 0; c < 256; ++c) {
    if (coding.sym_count[c] != 0) {
      Node node;
      node.count = coding.sym_count[c];
      node.sym = uint8_t(c);
      syms.push_back(node.sym);
      heap.push_back(node);
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
  get_code_len(tree, &heap[0], 0, coding.code_len);

  // Build "canonical Huffman code".

  // Sort the symbols in increasing order of code length using a counting sort.
  for (uint8_t sym : syms) {
    int len = coding.code_len[sym];
    ++coding.len_count[len];
    coding.len_mask |= 1 << len;
  }
  uint8_t cum_len_count[32] = {};
  for (int i = 0; i < 31; ++i) {
    cum_len_count[i + 1] = cum_len_count[i] + coding.len_count[i];
  }
  for (uint8_t sym : syms) {
    int len = coding.code_len[sym];
    coding.sorted_syms[cum_len_count[len]++] = sym;
  }
  coding.num_syms = syms.size();

  int current_len = 0;
  uint32_t current_code = 0;
  uint32_t current_inc = 1 << 31;
  for (int j = 0; j < coding.num_syms; ++j) {
    uint8_t sym = coding.sorted_syms[j];
    int len = coding.code_len[sym];
    if (current_len != len) {
      assert(current_len < len);
      // current_code <<= (len - current_len);
      current_len = len;
      current_inc = 1 << (32 - len);
    }
    assert(len >= 0);
    assert(len <= 32);
    coding.codes[sym].len = len;
    coding.codes[sym].bits = current_code;
#ifdef HUFF_DEBUG
    PrintCode(sym, current_code, len);
#endif
    current_code += current_inc;
  }
#ifdef HUFF_DEBUG
  std::cout << "compress: current_code at the end: " << current_code << "\n";
#endif
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
    cur_ |= uint64_t(code.bits) << (free_bits_ - 32);
    free_bits_ -= code.len;
    if (free_bits_ <= 32) {
      uint32_t to_write = htonl(cur_ >> 32);
      memcpy(output_, &to_write, 4);
      output_ += 4;
      free_bits_ += 32;
      cur_ <<= 32;
    }
  }

  void Finish() {
    while (free_bits_ < 64) {
      uint8_t top = cur_ >> 56;
      *output_ = top;
      ++output_;
      cur_ <<= 8;
      free_bits_ += 8;
    }
  }

 private:
  char* output_;
  uint64_t cur_;
  uint64_t free_bits_;
};

class CodeReader {
 public:
  CodeReader() : input_(nullptr), end_(nullptr), buf_bits_(0), buf_len_(0) {
    ConsumeBits(0);
  }

  CodeReader(const char* input, size_t size)
      : input_(input), end_(input + size), buf_bits_(0), buf_len_(0) {
    ConsumeBits(0);
  }

  void Init(const char* input, size_t size) {
    input_ = input;
    end_ = input + size;
    buf_bits_ = 0;
    buf_len_ = 0;
    ConsumeBits(0);
  }

  uint32_t GetTopBits() const {
    return (buf_bits_ >> (buf_len_ - 32)) & 0xFfffFfff;
  }

  void ConsumeBits(int num_bits) {
    buf_len_ -= num_bits;
    if (buf_len_ < 32) {
      if (input_ + 4 > end_) {
        // Special handling for the last bytes of the stream.
        // All further reads produce zeros.
        int num_bytes = end_ - input_;
        uint32_t to_add = 0;
        if (num_bytes != 0) {
          memcpy(&to_add, input_, num_bytes);
        }
        buf_bits_ <<= 32;
        buf_bits_ |= ntohl(to_add);
        buf_len_ += 32;
        input_ += num_bytes;
      } else {
        uint32_t to_add;
        memcpy(&to_add, input_, 4);
        buf_bits_ <<= 32;
        buf_bits_ |= ntohl(to_add);
        buf_len_ += 32;
        input_ += 4;
      }
    }
  }

 private:
  const char* input_;
  const char* end_;
  uint64_t buf_bits_;
  int buf_len_;
};

class Decoder {
 public:
  Decoder(uint8_t* len_count, const uint8_t* syms, int num_syms) {
    // Note that this code also handles the strange case where there is only 1
    // symbol in the compressed text, in which case that symbol is encoded
    // using 0 bits.
    int current_len = 0;
    int current_len_count = 0;
    uint32_t current_code = 0;
    uint64_t current_inc = 1ull << 32;
    // std::cout << "num_syms: " << num_syms << "\n" << std::flush;
    for (int i = 0; i < num_syms; ++i) {
      while (len_count[current_len] == current_len_count) {
        ++current_len;
        current_len_count = 0;
        current_inc = 1ull << (32 - current_len);
        // Make sure there are no holes in this array.
        code_begin_[current_len] = current_code;
      }
      if (len_syms_[current_len] == nullptr) {
        len_syms_[current_len] = &syms[i];
        code_begin_[current_len] = current_code;
      }
      assert(current_len < 32);

      if (current_len <= 8) {
        uint32_t x_begin = current_code >> 24;
        for (uint32_t x = x_begin; x < x_begin + (current_inc >> 24); ++x) {
          assert(x < 256);
          len_begin_[x] = current_len;
          fast_sym_[x] = syms[i];
        }
      } else {
        uint32_t x = current_code >> 24;
        if (len_begin_[x] == 0) {
          len_begin_[x] = current_len;
        }
      }
      current_code += current_inc;
      ++current_len_count;
    }
    // Should have exactly wrapped around:
    assert(current_code == 0);
  }

  int Decode(uint32_t code, uint8_t* out_sym) const {
    const int top_byte = code >> 24;
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
  }

 private:
  const uint8_t* len_syms_[32] = {};
  uint32_t code_begin_[32] = {};
  uint8_t len_begin_[256] = {};
  uint8_t fast_sym_[256] = {};
};

int CountBits(uint64_t x) { return __builtin_popcountll(x); }

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
  write_u32(compressed, coding.len_mask);
  for (uint8_t count : coding.len_count) {
    if (count != 0) {
      compressed.push_back(count);
    }
  }
  compressed.append(reinterpret_cast<char*>(coding.sorted_syms),
                    coding.num_syms);
  const int header_size = compressed.size();
  compressed.resize(header_size + (output_bits + 7) / 8);

  CodeWriter writer(&compressed[header_size]);
  for (uint8_t s : raw) {
    BitCode code = coding.codes[s];
    writer.WriteCode(code);
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
  Decoder decoder(
      len_count, reinterpret_cast<const uint8_t*>(compressed.data()), num_syms);
  compressed.remove_prefix(num_syms);

  std::string raw(raw_size, 0);
  CodeReader reader(compressed.data(), compressed.size());
  for (size_t i = 0; i < raw_size; ++i) {
    const uint32_t code = reader.GetTopBits();
    int bits_read = decoder.Decode(code, reinterpret_cast<uint8_t*>(&raw[i]));
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
  // Compute starting positions for each part.
  int end_offset[K] = {};
  {
    int pos = 0;
    for (int part = 0; part < K; ++part) {
      int64_t num_bits = 0;
      for (int i = 0; i < sizes[part]; ++i) {
        num_bits += coding.code_len[uint8_t(raw[pos + i])];
      }
      pos += sizes[part];
      end_offset[part] = (num_bits + 7) / 8;
    }
  }
  for (int i = 1; i < K; ++i) {
    end_offset[i] += end_offset[i - 1];
  }
  // TODO: Use varints
  const size_t header_size =
      4 + 4 + CountBits(coding.len_mask) + coding.num_syms + (K - 1) * 4;
  const size_t compressed_size = header_size + end_offset[K - 1];
  std::string compressed;
  compressed.reserve(compressed_size);
  write_u32(compressed, raw.size());
  write_u32(compressed, coding.len_mask);
  for (uint8_t count : coding.len_count) {
    if (count != 0) {
      compressed.push_back(count);
    }
  }
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
  const char* part_input[K];
  part_input[0] = raw.data();
  for (int i = 1; i < K; ++i) {
    part_input[i] = part_input[i - 1] + sizes[i - 1];
  }
  CodeWriter writer[K];
  for (int k = 0; k < K; ++k) {
    writer[k].Init(part_output[k]);
  }

  for (int i = 0; i < sizes[K - 1]; ++i) {
    for (int k = 0; k < K; ++k) {
      uint8_t sym = part_input[k][i];
      BitCode code = coding.codes[sym];
      writer[k].WriteCode(code);
    }
  }
  // Write potential last symbols.
  for (int k = 0; k < K; ++k) {
    if (sizes[k] > sizes[K - 1]) {
      uint8_t sym = part_input[k][sizes[k] - 1];
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
  char* part_output[K];
  part_output[0] = raw.data();
  for (int i = 1; i < K; ++i) {
    part_output[i] = part_output[i - 1] + sizes[i - 1];
  }

  for (int i = 0; i < sizes[K - 1]; ++i) {
    for (int k = 0; k < K; ++k) {
      const uint32_t code = reader[k].GetTopBits();
      int bits_read =
          decoder.Decode(code, reinterpret_cast<uint8_t*>(&part_output[k][i]));
      reader[k].ConsumeBits(bits_read);
    }
  }
  // Read last symbols.
  for (int k = 0; k < K; ++k) {
    if (sizes[k] > sizes[K - 1]) {
      const uint32_t code = reader[k].GetTopBits();
      uint8_t sym;
      int bits_read = decoder.Decode(code, &sym);
      reader[k].ConsumeBits(bits_read);
      part_output[k][sizes[k] - 1] = sym;
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
