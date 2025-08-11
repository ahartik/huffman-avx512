#include "codec/huffman.h"

#include <cassert>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <format>
#include <algorithm>
#include <memory>
#include <bit>
#include <vector>

namespace huffman {
namespace {

struct BitCode {
  uint32_t bits;
  int len;
};

void PrintCode(char sym, uint32_t bits, int len) {
  std::cout << "Sym: '" << sym << "', code: ";
  if (len < 0 || len > 32) {
    std::cout << "BAD LEN: " << len << "\n";
    return;
  }
  for (int i = 0; i < len; ++i) {
    std::cout << ((bits >> (31 - i)) & 1);
  }
  std::cout << "\n";
}

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

std::string compress(std::string_view raw) {
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
    PrintCode(sym, current_code, len);
    current_code += current_inc;

    ++len_count[len];
    len_mask |= 1 << len;
  }
  std::cout << "compress: current_code at the end: " << current_code << "\n";

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

    PrintCode(s, code.bits, code.len);
    // std::cout << std::format("code.bits: {:032B}\n", code.bits);
    std::cout << std::format("Cur: {:064B}\n", cur);

    if (free_bits <= 32) {
      write_u32(compressed, cur >> 32);
      std::cout << std::format("Wrote: {:032B}\n", cur >> 32);
      free_bits += 32;
      cur <<= 32;
    }
  }
  write_u32(compressed, cur >> 32);
  std::cout << std::format("Leftover write: {:032B}\n", cur >> 32);
  write_u32(compressed, 0);

  return compressed;
}

std::string decompress(std::string_view compressed) {
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
  std::vector<uint8_t> syms(
      reinterpret_cast<const uint8_t*>(&compressed[0]),
      reinterpret_cast<const uint8_t*>(&compressed[num_syms])
      );
  compressed.remove_prefix(num_syms);

  int current_len = 1;
  int current_len_count = 0;
  uint32_t current_code = 0;
  uint32_t current_inc = 1 << 31;
  uint8_t* len_syms[32] = {};
  uint32_t code_begin[32] = {};
  for (size_t i = 0; i < syms.size(); ++i) {
    const uint8_t sym = syms[i];
    while (len_count[current_len] == current_len_count) {
      ++current_len;
      current_len_count = 0;
      current_inc >>= 1;
    }
    if (len_syms[current_len] == nullptr) {
      len_syms[current_len] = &syms[i];
      code_begin[current_len] = current_code;
    }
    assert(current_len < 32);
    PrintCode(sym, current_code, current_len);

    current_code += current_inc;
    ++current_len_count;
  }
  for (int i = 0; i < 32; ++i) {
    std::cout << std::format("code_begin[{}] = {:032B}\n", i, code_begin[i]);
  }
  const int max_len = current_len;
  uint64_t buf_bits  = 0;
  int buf_len = 0;
  std::string raw(raw_size, 0);
  for (size_t i = 0; i < raw_size; ++i) {
    if (buf_len < 32) {
      buf_bits <<= 32;
      buf_bits |= read_u32(compressed);
      buf_len += 32;
    }
    std::cout << std::format("Buf {:064B} len = {}\n", buf_bits, buf_len);
    // TODO: Optimize
    uint32_t code = (buf_bits >> (buf_len - 32))& 0xfFFFfFFF;
    int len = max_len;
    for (int j = 1; j <= max_len; ++j) {
      if (code_begin[j] > code) {
        len = j-1;
        break;
      }
    }
    std::cout << "code len: " << len << "\n";
    int offset = (code - code_begin[len]) >> (32 - len);
    uint8_t sym = len_syms[len][offset];
    raw[i] = sym;
    buf_len -= len;
    std::cout << std::format("Read {:032B} -> '{}'\n", code, char(sym));
  }
  return raw; 
}

}  // namespace huffman
