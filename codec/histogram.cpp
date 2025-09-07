#include "codec/histogram.h"

#include <immintrin.h>
#include <x86intrin.h>
#include <cassert>
#include <cstring>

#include <memory>

#define UNROLL8 _Pragma("GCC unroll 8")

namespace huffman {

ByteHistogram MakeHistogramVectorized(std::string_view text) {
  // I think the () at the end of the new-expression guarantees that the array
  // gets zeroed.
  ByteHistogram sym_count = {};
  constexpr int NUM_ARR = 4;
  const std::unique_ptr<std::array<uint32_t, 256>[]> tmp_count(
      new std::array<uint32_t, 256>[NUM_ARR]());
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
#if 0
#define ADD_ONE(j)                         \
  do {                                     \
    uint64_t b = _mm_extract_epi8(vec, j); \
    ++tmp_count[j % NUM_ARR][b];           \
  } while (0)

      // clang-format off
      ADD_ONE(0); ADD_ONE(1); ADD_ONE(2); ADD_ONE(3);
      ADD_ONE(4); ADD_ONE(5); ADD_ONE(6); ADD_ONE(7);
      ADD_ONE(8); ADD_ONE(9); ADD_ONE(10); ADD_ONE(11);
      ADD_ONE(12); ADD_ONE(13); ADD_ONE(14); ADD_ONE(15);
      // clang-format on
#undef ADD_ONE
#elif 1

#define ADD_TWO(j)                                       \
  do {                                                   \
    uint16_t b = _mm_extract_epi16(vec, j);              \
    ++tmp_count[(j * 2) % NUM_ARR][b & 0xff];            \
    ++tmp_count[(j * 2 + 1) % NUM_ARR][(b >> 8) & 0xff]; \
  } while (0)

      // clang-format off
      ADD_TWO(0); ADD_TWO(1); ADD_TWO(2); ADD_TWO(3);
      ADD_TWO(4); ADD_TWO(5); ADD_TWO(6); ADD_TWO(7);
      // clang-format on
#undef ADD_TWO

#else

#define ADD_FOUR(j)                           \
  do {                                        \
    uint32_t b = _mm_extract_epi32(vec, j);   \
    ++tmp_count[j * 4][b & 0xff];             \
    ++tmp_count[j * 4 + 1][(b >> 8) & 0xff];  \
    ++tmp_count[j * 4 + 2][(b >> 16) & 0xff]; \
    ++tmp_count[j * 4 + 3][(b >> 24) & 0xff]; \
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
    UNROLL8 for (int j = 0; j < NUM_ARR; ++j) {
      sym_count[c] += tmp_count[j][c];
    }
  }
  return sym_count;
}

// This is not very fast :(
ByteHistogram MakeHistogramGatherScatter(std::string_view text) {
  // I think the () at the end of the new-expression guarantees that the array
  // gets zeroed.
  ByteHistogram sym_count;
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
  if (ptr + 16 < end) {
    __m128i cached = _mm_loadu_si128((const __m128i*)ptr);
    ptr += 16;
    while (ptr + 16 < end) {
      __m128i bytes = cached;
      cached = _mm_loadu_si128((const __m128i*)ptr);
      ptr += 16;
      __m512i index = _mm512_cvtepu8_epi32(bytes);
      index = _mm512_add_epi32(index, offset_v);

      __m512i cnt = _mm512_i32gather_epi32(index, tmp_count.get(), 4);
      cnt = _mm512_add_epi32(cnt, one32);

      _mm512_i32scatter_epi32(tmp_count.get(), index, cnt, 4);
    }
    ptr -= 16;
  }

  while (ptr < end) {
    ++tmp_count[*ptr++];
  }
  for (int c = 0; c < 256; ++c) {
    sym_count[c] = 0;
    UNROLL8 for (int j = 0; j < 16; ++j) {
      sym_count[c] += tmp_count[j * 256 + c];
    }
  }
  return sym_count;
}

ByteHistogram MakeHistogramMulti(std::string_view text) {
  ByteHistogram hist = {};
  // It's not exactly clear if 64 or 32 bits is faster here.
  using WordType = uint64_t;
  constexpr int WORD_BYTES = sizeof(WordType);
  // Using 4 arrays is faster when the distribution is uniform, but slower
  // when the distribution is biased.
  constexpr int NUM_ARR = 8;
  uint32_t tmp_count[NUM_ARR][256] = {};
  const size_t text_size = text.size();
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(text.data());
  const uint8_t* end = ptr + text_size;

  if (ptr + WORD_BYTES - 1 < end) {
    WordType cached;
    memcpy(&cached, ptr, WORD_BYTES);
    ptr += WORD_BYTES;
    // Unrolling here didn't seem to make it faster.
    while (ptr + WORD_BYTES - 1 < end) {
      WordType data = cached;
      memcpy(&cached, ptr, WORD_BYTES);
      ptr += WORD_BYTES;
      UNROLL8 for (int j = 0; j < WORD_BYTES; ++j) {
        ++tmp_count[j % NUM_ARR][(data >> (j * 8)) & 0xff];
      }
    }
    ptr -= WORD_BYTES;
  }
  while (ptr < end) {
    ++tmp_count[0][*ptr++];
  }
  for (int c = 0; c < 256; ++c) {
    hist[c] = 0;
    for (int j = 0; j < NUM_ARR; ++j) {
      hist[c] += tmp_count[j][c];
    }
  }
  return hist;
}

ByteHistogram MakeHistogramSimple(std::string_view text) {
  ByteHistogram hist = {};
  const size_t text_size = text.size();
  for (size_t i = 0; i < text_size; ++i) {
    ++hist[uint8_t(text[i])];
  }
  return hist;
}

ByteHistogram MakeHistogram(std::string_view text) {
  // Idea copied from Huff0: count in multiple arrays to maximize
  // instructions per cycle (superscalar).
  if (text.size() < 1500) {
    return MakeHistogramSimple(text);
  } else {
    return MakeHistogramMulti(text);
  }
}
}  // namespace huffman
