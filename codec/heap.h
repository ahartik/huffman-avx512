#pragma once

#include <immintrin.h>
#include <x86intrin.h>
#include <cassert>
#include <cstdint>

#include <algorithm>
#include <limits>

namespace huffman {

class SimdHeap {
 public:
  static constexpr uint32_t kMaxCount = (1 << 24) - 1;
  static constexpr uint32_t kMaxSize = 256;
  static constexpr uint32_t kEmpty = std::numeric_limits<uint32_t>::max();
  SimdHeap() {
    size_ = 0;
    std::fill(count_, count_ + 256, kEmpty);
    std::fill(block_min_, block_min_ + 16, kEmpty);
  };

  void AddInitial(uint32_t count, int32_t data) {
    assert(count < kMaxCount);
    assert(size_ < 256);
    count_[size_] = (count << 8) | size_;
    data_[size_] = data;
    ++size_;
  }

  void Init() {
    for (int j = 0; j < 16; ++j) {
      block_min_[j] = GetMin16(count_ + j * 16);
    }

    // Initialize freelist.
    next_free_ = size_;
    for (int i = size_; i < kMaxSize; ++i) {
      data_[i] = i + 1;
    }
  }

  void Push(uint32_t count, int32_t data) {
    assert(count < kMaxCount);
    assert(size_ < 256);

    // Find spot from freelist.
    int ind = next_free_;
    assert(ind < 256);
    next_free_ = data_[next_free_];
    count_[ind] = (count << 8) | ind;
    data_[ind] = data;

    // Update this block:
    uint32_t block = ind / 16;
    block_min_[block] = GetMin16(count_ + 16 * block);

    ++size_;
  }

  struct CountAndData {
    uint32_t count;
    int32_t data;
  };

  CountAndData  __attribute__ ((noinline)) 
  Pop() {
    const uint32_t min_val = GetMin16(block_min_);
    const uint32_t min_ind = min_val & 0xff;

    CountAndData popped = {
        .count = (min_val >> 8),
        .data = data_[min_ind],
    };
    --size_;

    count_[min_ind] = kEmpty;
    // Make `min_ind` the head of the freelist.
    data_[min_ind] = next_free_;
    next_free_ = min_ind;

    unsigned pop_block = min_ind / 16;
    block_min_[pop_block] = GetMin16(count_ + 16 * pop_block);

    return popped;
  }

  // TODO: At least one call to GetMin16 can be removed for one loop of Huffman
  // code construction: It always performs two Pop()s followed by a Push().
  //
  // After the second pop, we don't need to update 
  // 
  // 1. Find minimum item and fill that spot with kEmpty
  // 2. Update block_min_ for that block
  // 3. Find second 

  size_t size() const { return size_; }

 private:
  uint32_t GetMin16(uint32_t* arr) {
    // TODO: This could potentially be optimized too.
    return _mm512_reduce_min_epu32(_mm512_loadu_epi32(arr));
  }
  uint32_t count_[kMaxSize];
  uint32_t block_min_[16];
  int32_t data_[kMaxSize];
  size_t size_;
  int next_free_;
};

class BinaryHeap {
 public:
  static constexpr uint32_t kMaxCount = (1 << 24) - 1;
  static constexpr uint32_t kMaxSize = 256;
  static constexpr uint32_t kEmpty = std::numeric_limits<uint32_t>::max();

  BinaryHeap() { size_ = 0; };

  void AddInitial(uint32_t count, int32_t data) {
    heap_[size_] = {.count = count, .data = data};
    ++size_;
  }

  void Init() { std::make_heap(heap_, heap_ + size_); }

  void Push(uint32_t count, int32_t data) {
    AddInitial(count, data);
    std::push_heap(heap_, heap_ + size_);
  }

  struct CountAndData {
    uint32_t count;
    int32_t data;
  };

  CountAndData Pop() {
    CountAndData popped = {
        .count = heap_[0].count,
        .data = heap_[0].data,
    };
    std::pop_heap(heap_, heap_ + size_);
    --size_;
    return popped;
  }

  size_t size() const { return size_; }

 private:
  struct Node {
    uint32_t count;
    int32_t data;
    bool operator<(const Node& o) const { return count > o.count; }
  };
  Node heap_[kMaxSize];
  size_t size_;
};

}  // namespace huffman
