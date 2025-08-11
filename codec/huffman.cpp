#include "codec/huffman.h"

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
    return std::format("{:c}",sym);
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

  // TODO: Modify this to not require so many dynamic allocations.
  std::unique_ptr<Node> children[2];
  uint8_t sym = 0;
  // Opposite order, since C++ heap is a max heap and we want
  // to pop smallest counts first.
  bool operator<(const Node& o) const { return count > o.count; }
};

void get_code_len(const Node* node, int len, int* code_len) {
  if (node->children[0] == nullptr) {
    code_len[node->sym] = len;
  } else {
    get_code_len(node->children[1].get(), len + 1, code_len);
    get_code_len(node->children[0].get(), len + 1, code_len);
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

}  // namespace

std::string Compress(std::string_view raw) {
  int count[256] = {};
  for (const unsigned char c : raw) {
    ++count[c];
  }
  std::vector<Node> heap;
  std::vector<uint8_t> syms;
  for (int c = 0; c < 256; ++c) {
    if (count[c] != 0) {
      Node node;
      node.count = count[c];
      node.sym = uint8_t(c);
      syms.push_back(node.sym);
      heap.push_back(std::move(node));
    }
  }
  std::make_heap(heap.begin(), heap.end());
  while (heap.size() > 1) {
    // Pop two elements
    Node a = std::move(heap[0]);
    std::pop_heap(heap.begin(), heap.end());
    heap.pop_back();
    Node b = std::move(heap[0]);
    std::pop_heap(heap.begin(), heap.end());
    heap.pop_back();

    Node next;
    next.count = a.count + b.count;
    next.children[0] = std::make_unique<Node>(std::move(a));
    next.children[1] = std::make_unique<Node>(std::move(b));
    heap.push_back(std::move(next));
    std::push_heap(heap.begin(), heap.end());
  }
  int code_len[256] = {};
  get_code_len(&heap[0], 0, code_len);

  // Build "canonical Huffman code".
  std::sort(syms.begin(), syms.end(), [&code_len](uint8_t a, uint8_t b) {
    if (code_len[a] == code_len[b]) {
      return a < b;
    }
    return code_len[a] < code_len[b];
  });
  BitCode codes[256];
  int current_len = 1;
  uint32_t current_code = 0;
  uint32_t current_inc = 1 << 31;
  uint8_t len_count[32] = {};
  uint32_t len_mask = 0;
  for (uint8_t sym : syms) {
    int len = code_len[sym];
    if (current_len != len) {
      // current_code <<= (len - current_len);
      current_len = len;
      current_inc = 1 << (32 - len);
    }
    assert(len > 0);
    assert(len < 32);
    codes[sym].len = len;
    codes[sym].bits = current_code;
#ifdef HUFF_DEBUG
    PrintCode(sym, current_code, len);
#endif
    current_code += current_inc;

    ++len_count[len];
    len_mask |= 1 << len;
  }
#ifdef HUFF_DEBUG
  std::cout << "compress: current_code at the end: " << current_code << "\n";
#endif

  std::string compressed;
  // TODO: Fail for too long strings.
  write_u32(compressed, raw.size());
  write_u32(compressed, len_mask);
  for (uint8_t count : len_count) {
    if (count != 0) {
      compressed.push_back(count);
    }
  }
  for (uint8_t sym : syms) {
    compressed.push_back(sym);
  }

  uint64_t cur = 0;
  uint64_t free_bits = 64;
  for (uint8_t s : raw) {
    BitCode code = codes[s];
    cur |= uint64_t(code.bits) << (free_bits - 32);
    free_bits -= code.len;

#ifdef HUFF_DEBUG
    PrintCode(s, code.bits, code.len);
    // std::cout << std::format("code.bits: {:032B}\n", code.bits);
    std::cout << std::format("Cur: {:064B}\n", cur);
#endif

    if (free_bits <= 32) {
      write_u32(compressed, cur >> 32);
#ifdef HUFF_DEBUG
      std::cout << std::format("Wrote: {:032B}\n", cur >> 32);
#endif
      free_bits += 32;
      cur <<= 32;
    }
  }
  write_u32(compressed, cur >> 32);
#ifdef HUFF_DEBUG
  std::cout << std::format("Leftover write: {:032B}\n", cur >> 32);
#endif
  write_u32(compressed, 0);

  return compressed;
}

std::string Decompress(std::string_view compressed) {
  // Build codebook.
  const uint32_t raw_size = read_u32(compressed);
  const uint32_t len_mask = read_u32(compressed);
  uint8_t len_count[32] = {};
  int num_syms = 0;
  int lens[32] = {};
  int num_lens = 0;
  for (int i = 0; i < 32; ++i) {
    if (len_mask & (1 << i)) {
      len_count[i] = uint8_t(compressed[0]);
      compressed.remove_prefix(1);
      num_syms += len_count[i];

      lens[num_lens] = i;
      ++num_lens;
    }
  }
  std::vector<uint8_t> syms(
      reinterpret_cast<const uint8_t*>(&compressed[0]),
      reinterpret_cast<const uint8_t*>(&compressed[num_syms]));
  compressed.remove_prefix(num_syms);

  int current_len = 1;
  int current_len_count = 0;
  uint32_t current_code = 0;
  uint32_t current_inc = 1 << 31;
  uint8_t* len_syms[32] = {};
  uint32_t code_begin[32] = {};

  // Build a map from first 8 bits of the code to code length.
  uint8_t len_begin[256] = {};
  // For short codes, we immediately know the output symbol.
  uint8_t fast_sym[256] = {};

#ifdef HUFF_DEBUG
  uint8_t code_len[256] = {};
#endif
  for (size_t i = 0; i < syms.size(); ++i) {
    const uint8_t sym = syms[i];
    while (len_count[current_len] == current_len_count) {
      ++current_len;
      current_len_count = 0;
      current_inc >>= 1;
      // Make sure there are no holes in this array.
      code_begin[current_len] = current_code;
    }
    if (len_syms[current_len] == nullptr) {
      len_syms[current_len] = &syms[i];
      code_begin[current_len] = current_code;
    }
    assert(current_len < 32);
#ifdef HUFF_DEBUG
    PrintCode(sym, current_code, current_len);
    code_len[sym] = current_len;
#endif

    if (current_len <= 8) {
      uint32_t x_begin = current_code >> 24;
      for (uint32_t x = x_begin; x < x_begin + (current_inc >> 24); ++x) {
        len_begin[x] = current_len;
        fast_sym[x] = sym;
      }
    } else {
      uint32_t x = current_code >> 24;
      if (len_begin[x] == 0) {
        len_begin[x] = current_len;
      }
    }
    current_code += current_inc;
    ++current_len_count;
  }
  // Should have exactly wrapped around:
  assert(current_code == 0);
  assert(len_begin[255] >= 0);
#ifdef HUFF_DEBUG
  for (int i = 0; i < 32; ++i) {
    std::cout << std::format("code_begin[{:2d}] = {:032B}\n", i, code_begin[i]);
  }
  for (int i = 0; i < 256; ++i) {
    if (len_begin[i] == 0) {
      std::cout << "i=" << i << "\n" << std::flush;
    }
    assert(len_begin[i] != 0);
  }
#endif

  const int max_len = current_len;
  uint64_t buf_bits = 0;
  int buf_len = 0;
  std::string raw(raw_size, 0);
  for (size_t i = 0; i < raw_size; ++i) {
    if (buf_len < 32) {
      buf_bits <<= 32;
      buf_bits |= read_u32(compressed);
      buf_len += 32;
    }
#ifdef HUFF_DEBUG
    std::cout << std::format("Buf {:064B} len = {}\n", buf_bits, buf_len);
#endif

    // TODO: Optimize
    const uint32_t code = (buf_bits >> (buf_len - 32)) & 0xfFFFfFFF;

    const int top_byte = code >> 24;
    int len = len_begin[top_byte];
#if 0
    if (len <= 8) {
      // Fast path, know the symbol based on the top 8 bits.
      raw[i] = fast_sym[top_byte];
      buf_len -= len;
    } else
#endif
    {
      // We don't know the exact length of this code
      while ((code_begin[len + 1] != 0) & (code_begin[len + 1] <= code)) {
        ++len;
      }
#ifdef HUFF_DEBUG
      std::cout << "code len: " << len << "\n" << std::flush;
#endif
      int offset = (code - code_begin[len]) >> (32 - len);
      uint8_t sym = len_syms[len][offset];
      raw[i] = sym;
      buf_len -= len;
#ifdef HUFF_DEBUG
      std::cout << std::format("Read {:032B} -> '{}'\n", code, SymToStr(sym));
      std::cout << "code_len[sym] = " << int(code_len[sym]) << "\n"
                << std::flush;
      assert(len == code_len[sym]);
#endif
    }
  }
  return raw;
}

}  // namespace huffman
