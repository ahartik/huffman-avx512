# Introduction: Huffman coding and AVX-512 
[Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding) is 
a very commonly used method to represent a stream of symbols by using a
variable number of bits for each symbol based on their frequency. It is used by
popular formats such as .zip, .gz, .jpg, .png and more.

Zstandard is a very popular compression algorithm as of 2025,
because it offers a good combination of speed and and compression
efficiency.
One of the tools Zstandard uses to achieve good performance
is the used Huff0 Huffman coding.
Huff0 is fast because it takes advantage of the fact that modern CPUs can
execute more than one instruction for every cycle.
By splitting the data to four (4) streams, decompression
can perform many more instructions per cycle (IPC).
CPUs capable of executing more than one instruction per cycle
are called "superscalar".

Modern x86 CPUs, in addition to being able to perform multiple
instructions per cycle, have instructions that work on not just singular (i.e. scalar) values,
but on vectors of up to 512 bits in AVX-512.
The goal of this repository is to explore whether AVX-512 vector
instructions in particular can be used to speed up Huffman coding.


I wanted to explore whether AVX-512 instructions could make Huffman compression
or decompression faster.
The baseline I'm comparing against is the
[Huff0](https://github.com/Cyan4973/FiniteStateEntropy) codec by Yann Collet,
which is also used in the very popular
[Zstandard](https://github.com/facebook/zstd) compression algorithm.
(The code I used is not the assembly-optimized version found in modern
 Zstandard, but an earlier plain C version.)
One of the things I learned here is that Huff0 (and likely Zstandard) are fast
because the data format is built around making a fast decoder.
I embraced the same spirit here, looking for a data format which is fast
when using AVX-512.



# Results
The code was developed and tested using an AMD Ryzen 9950X CPU.

| Method             | Streams  | Compress MiB/s | Decompress MiB/s | IPC  |
|--------------      | -------- | -------------- | ---------------- | ---- |
| Huff0              |  4       |                |                  |
| scalar             |  1       |                |                  |
| scalar             |  4       |                |                  |
| AVX-512 gather     |  8       |                |                  |
| AVX-512 gather     |  32      |                |                  |
| AVX-512 register   |  8       |                |                  |
| AVX-512 register   |  32      |                |


Here "Single" is my implementation of regular Huffman coding, where 
My scalar code is slightly less well optimized than Huff0, resulting 

The bottleneck of the AVX-512 decompression is the `vpgatherqq` instruction
(`_mm512_i64_gather_epi64` intrinsic), which loads 8 64-bit integers from
memory using 64-bit offsets.
This instruction is implemented as "microcoded"
on AMD Zen 5, and according to [Agner's instruction
tables](https://www.agner.org/optimize/instruction_tables.pdf), it results in
40 [micro-ops (μops)](https://en.wikipedia.org/wiki/Micro-operation).
This explains the extremely low 

## Breakdown of compression

# Methods

## scalar

This method is simply an optimized, plain C++ implementation of Huffman
coding.
The implementation ended up quite closely following approaches used by Huff0.
This is because Huff0 is so fast, and after comparing my initial code with Huff0
I ended up trying and implementing some ideas found in Huff0 code, such as
two-symbols at a time decoding.

The performance is quite close to the 

## AVX-512 Gather

These implementations of compression and decompression are relatively
straight-forward vectorizations of the "scalar" method.
This means array lookups are converted to "gather" instructions,
which load elements from memory based on indices
in SIMD registers.

Notably, implementations of gather instructions are not
especially fast in today's AMD CPUs (Zen5).
Gather instructions are implemented as "microcoded"
on AMD Zen 5, and according to [Agner's instruction
tables](https://www.agner.org/optimize/instruction_tables.pdf), they result
in up to 40 [micro-ops (μops)](https://en.wikipedia.org/wiki/Micro-operation).

In the implementation 512-bit vectors are used to process 8 64-bit elements at
a time, each element corresponding to a data stream.
Multiples of 8 data streams are supported, leading to higher IPC.

Using 32 streams with AVX-512 gather method gives the best decompression speed.

## AVX-512 Permute

CPUs supporting AVX-512 have 32 512-bit registers (ZMM0 to ZMM31).
In the AVX-512 "register" method, I use these registers
as lookup tables instead of relying on gather instructions.
A key of this approach is the the VPERMB instruction included in
[AVX512\_VBMI](https://en.wikichip.org/wiki/x86/avx512_vbmi)
, which is supported today by AMD Zen4, AMD Zen5, and some more modern Intel
CPUs such as Cannon Lake, Ice Lake, and Sapphire Rapids.
This instruction can perform an arbitrary byte permutation,
meaning 512-bit registers can be treated as 64-value lookup tables for bytes.
Lookups to a 256-element table can be performed using four registers
and permute instructions masked with comparison results.

This approach gives the best compression speed, since compression is
essentially just a single lookup for each symbol from a 256-element array.
Four permutes and three comparisons are used to perform lookups for 64 symbols.
XXX: Not including symbol counting this is XX % faster than scalar code and
Huff0.

Using this approach in decompression, fully avoiding the gather instructions,
requires a more complex approach.

Decompression speed using this method is not as fast,
since the regular decoding tables cannot be fit in registers
and thus require a completely different approach.
To decode the bitstream using the register method, we find the current code
length (for each stream) using multiple comparisons,
and find the symbol index by subtraction.
This decompression method relies on the code being a canonical Huffman code.

This approach works particularly well for the compression,
XXX PERF

Decompression speed is a bit slower than multi-stream scalar compression (Huff0
and scalar/4), and XXX % slower than the lookup-table method using gather
instructions.

# Details

## Compression breakdown

### Symbol-counting

## Length-limited codes

## 

# Conclusions: what did we learn and is this at all useful?

Using AVX-512, we find compression to be XX % faster and decompression
to be XXYY% faster.
I find this a bit underwhelming,
given that each instruction is processing 8 times the amount of data compared
to the scalar algorithm.

Main reason for 

The speed results are slightly underwhelming. Given that we are processing 8 
times the amount of data with every instruction, one would hope
to achieve better than XX % 
