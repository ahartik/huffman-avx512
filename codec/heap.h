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
  };

  void AddInitial(uint32_t count, int32_t data) { Push(count, data); }
  void Init() {}

  void Push(uint32_t count, int32_t data) {
    assert(count < kMaxCount);
    assert(size_ < 256);
    count_[size_] = (count << 8) | size_;
    data_[size_] = data;
    ++size_;
  }

  struct CountAndData {
    uint32_t count;
    int32_t data;
  };

  CountAndData /* __attribute__ ((noinline)) */
  Pop() {
    using vec16x32 = __m512i;
    assert(size_ > 0);

    vec16x32 min16 = _mm512_loadu_epi32(count_);
    for (int j = 16; j < size_; j += 16) {
      min16 = _mm512_min_epu32(min16, _mm512_loadu_epi32(count_ + j));
    }

    const uint32_t min_val = _mm512_reduce_min_epu32(min16);
    const int min_ind = min_val & 0xff;

    CountAndData popped = {
        .count = (min_val >> 8),
        .data = data_[min_ind],
    };
    --size_;
    count_[min_ind] = (count_[size_] & 0xFFFF'FF00) | (min_ind);
    data_[min_ind] = data_[size_];
    count_[size_] = kEmpty;
    return popped;
  }

  size_t size() const { return size_; }

 private:
  uint32_t count_[kMaxSize];
  int32_t data_[kMaxSize];
  size_t size_;
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
