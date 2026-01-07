# Introduction: Huffman coding and AVX-512 
[Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding) is 
a very commonly used method to represent a stream of symbols by using a
variable number of bits for each symbol based on their frequency. It is used by
popular formats such as .zip, .gz, .jpg, .png and more.

Zstandard is a very popular compression algorithm as of 2025,
because it offers a good combination of speed and and compression
efficiency.
A key component of Zstandard is its
Huffman coding implementation, called Huff0.
Huff0 is fast because it takes advantage of the fact that modern CPUs can
execute more than one instruction every cycle.
By splitting the data to four (4) streams, decompression
achieves higher number of instructions per cycle (IPC).
CPUs capable of executing more than one instruction per cycle
are called "superscalar".
One of the first superscalar processors was the first Intel Pentium (P5).

Modern x86 CPUs, in addition to being able to perform multiple instructions per
cycle, have instructions that work on not just singular (i.e. scalar) values,
but on vectors of up to 512 bits in AVX-512.
The goal of this project was to explore whether AVX-512 vector
instructions in particular can be used to speed up Huffman coding.

I wanted to explore whether AVX-512 instructions could make Huffman compression
or decompression faster.
The baseline I'm comparing against is the
[Huff0](https://github.com/Cyan4973/FiniteStateEntropy) codec by Yann Collet,
which is also used in the very popular
[Zstandard](https://github.com/facebook/zstd) compression algorithm.
(The code I used is not the assembly-optimized version found in modern
 Zstandard, but an earlier plain C version.)
The data format of Huff0 is expertly designed such that the encoder is
as fast as possible.
I embraced the same spirit here, looking for a data format which is fast
when using AVX-512.


# Results
The code was developed and tested using an AMD Ryzen 9950X CPU.

## Biased input

Probability of symbol $i \in \{0, \ldots, 255\}$ is $p_i = 0.8^i \cdot 0.2$.

Method | Streams | Compress | Decompress
-------|--- | ---| ---
Scalar | 1 | **1947 MiB/s** | **1056 MiB/s**
Scalar | 4 | **1834 MiB/s** | **3616 MiB/s**
Scalar | 8 | 1952 MiB/s | **3218 MiB/s**
AVX-512 Gather | 8 | 1610 MiB/s | 1842 MiB/s
AVX-512 Permute | 8 | **2637 MiB/s** | 1162 MiB/s
AVX-512 Gather | 16 | 1904 MiB/s | **3355 MiB/s** 
AVX-512 Permute | 16 | **2988 MiB/s** | 2123 MiB/s
AVX-512 Gather | 32 | 1858 MiB/s | **5026 MiB/s**
AVX-512 Permute | 32 | **2926 MiB/s** | 3198 MiB/s
AVX-512 Gather | 48 | 1674 MiB/s | **4950 MiB/s**
AVX-512 Permute | 48 | **2673 MiB/s** | 3453 MiB/s
Huff0 | 4 | 1946 MiB/s | 3636 MiB/s

Using the fastest AVX-512 methods results with 32 streams results
in 50% faster compression and 38% faster decompression
compared to Huff0.

## English Wikipedia: 

First 100 KiB of [enwik8](https://mattmahoney.net/dc/textdata.html).

Method | Streams | Compress | Decompress
-------|--- | ---| ---
Scalar | 1 | 1890 MiB/s | 972 MiB/s
Scalar | 4 | 1782 MiB/s | 2953 MiB/s
Scalar | 8 | 1879 MiB/s | 2625 MiB/s
AVX-512 Gather | 8 | 1559 MiB/s | 1596 MiB/s
AVX-512 Permute | 8 | 2498 MiB/s | 1155 MiB/s
AVX-512 Gather | 16 | 1837 MiB/s | 2803 MiB/s
AVX-512 Permute | 16 | 2854 MiB/s | 2109 MiB/s
AVX-512 Gather | 32 | 1822 MiB/s | 4039 MiB/s
AVX-512 Permute | 32 | 2803 MiB/s | 3177 MiB/s
AVX-512 Gather | 48 | 1651 MiB/s | 3994 MiB/s
AVX-512 Permute | 48 | 2641 MiB/s | 3441 MiB/s
Huff0 | 4 | 1956 MiB/s | 2974 MiB/s

Using the fastest AVX-512 methods results with 32 streams results
in 43% faster compression and 36% faster decompression
compared to Huff0.

# Methods and analysis

Besides Huff0, the data format is equivalent between scalar and AVX code
for the same number of streams.

## Scalar

These methods were my implementations of Huffman coding split to multiple
streams while sticking to regular scalar code.
Initially these werent very fast, so I studied Huff0 code and copied many ideas
and tricks.
Unlike Huff0, my code supports a template-parameterized number of streams,
which allows comparison between different stream count.


## AVX Gather

These methods are straightforward vectorizations of the scalar method, with
reads and writes replaced by gather and scatter instructions respectively.

On AMD Zen 5, gather and scatter instructions are implemented as "microcoded".
According to [Agner's instruction
tables](https://www.agner.org/optimize/instruction_tables.pdf),
the used gather instruction (`vpgatherqq`/`_mm512_i64_gather_epi64`) 
results in
40 [micro-ops (μops)](https://en.wikipedia.org/wiki/Micro-operation).


##  AVX Permute

These methods use AVX-512 registers instead of lookup tables for encoding and
decoding.
An array of 256 bytes can be stored in four registers, allowing 64 lookups
to be performed using two `vpermi2b`/ byte permute instructions.

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
