# Using AVX-512 for Huffman coding

[Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding) is 
a very commonly used method to represent a stream of symbols by using a
variable number of bits for each symbol based on their frequency. It is used by
popular formats such as .zip, .gz, .jpg, .png and more.

I wanted to explore whether AVX-512 instructions could make Huffman compression
or decompression faster.
The baseline I'm comparing against is the
[Huff0](https://github.com/Cyan4973/FiniteStateEntropy) codec by Yann Collet,
which is also used in the very popular
[Zstandard](https://github.com/facebook/zstd) compression algorithm.
(The code I used however is not the assembly-optimized version found in modern
 Zstandard.)


# Results
The code was developed and tested using an AMD Ryzen 9950X CPU.

| Method       | Compress MiB/s | Decompress MiB/s | IPC    |
|--------------| -------- | ---------- | ------ |
| Huff0        |
| Single       |
| Multi4       |
| AvxMulti8    |
| AvxMulti32   |

Here "Single" is my implementation of regular Huffman coding, where 
My scalar code is slightly less well optimized than Huff0, resulting 

The bottleneck of the AVX-512 decompression is the `vpgatherqq` instruction
(`_mm512_i64_gather_epi64` intrinsic), which loads 8 64-bit integers from
memory using 64-bit offsets. This instruction is implemented as "microcoded"
on AMD Zen 5, and according to [Agner's instruction
tables](https://www.agner.org/optimize/instruction_tables.pdf), it results in
40 [micro-ops (μops)](https://en.wikipedia.org/wiki/Micro-operation).
This explains the extremely low 


