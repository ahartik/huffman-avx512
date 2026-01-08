# Huffman coding using AVX-512 
[Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding) is 
a popular method to represent a stream of symbols using a
variable number of bits for each symbol based on symbol frequency.
It is used by popular formats such as .zip, .gz, .jpg, .png and more.

[Zstandard](https://github.com/facebook/zstd)
by Yann Collet is a compression algorithm offering very fast speed and decent
compression efficiency.
It is one of the most popular compression algorithms as of 2026.
A key component of Zstandard is its Huffman coding implementation, called Huff0.
Huff0 is fast because it takes advantage of
the fact that modern CPUs can
execute more than one instruction every cycle.
By splitting the data to four (4) streams, decompression
achieves higher number of instructions per cycle (IPC).
CPUs capable of executing more than one instruction per cycle
are called "superscalar", with original Intel Pentium (P5) being
one of the first superscalar CPU.

Modern x86-64 CPUs, in addition to being able to perform multiple instructions per
cycle, have instructions that work on not just singular (i.e. scalar) values,
but on vectors of up to 512 bits in AVX-512.
The goal of this project is to explore whether AVX-512 vector
instructions in particular can be used to speed up Huffman coding.

The baseline I'm comparing against is the
[Huff0](https://github.com/Cyan4973/FiniteStateEntropy) code by Yann Collet.
The data format of Huff0 is expertly designed such that the encoder is
as fast as possible.
I embraced the same spirit here, looking for a data format which is fast
when using AVX-512.

# Results
The code was developed and tested using an AMD Ryzen 9950X CPU.

## Biased input

Input is random-generated, with probability of symbol
$i \in \{0, \ldots, 255\}$ being $p_i = 0.8^i \cdot 0.2$.

Fastest method for each stream count is bolded, not including Huff0.

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
Scalar | 1 | **1890 MiB/s** | **972 MiB/s**
Scalar | 4 | **1782 MiB/s** | **2953 MiB/s**
Scalar | 8 | 1879 MiB/s | **2625 MiB/s**
AVX-512 Gather | 8 | 1559 MiB/s | 1596 MiB/s
AVX-512 Permute | 8 | **2498 MiB/s**| 1155 MiB/s
AVX-512 Gather | 16 | 1837 MiB/s | **2803 MiB/s**
AVX-512 Permute | 16 | **2854 MiB/s** | 2109 MiB/s
AVX-512 Gather | 32 | 1822 MiB/s | **4039 MiB/s**
AVX-512 Permute | 32 | **2803 MiB/s** | 3177 MiB/s
AVX-512 Gather | 48 | 1651 MiB/s | **3994 MiB/s**
AVX-512 Permute | 48 | **2641 MiB/s** | 3441 MiB/s
Huff0 | 4 | 1956 MiB/s | 2974 MiB/s

Using the fastest AVX-512 methods results with 32 streams results
in 43% faster compression and 36% faster decompression
compared to Huff0.

# Methods and analysis

All methods except Huff0 are my code.
The data format of **scalar** and **AVX-512** methods is heavily inspired
by Huff0, but not the same. The data format of these methods is equal
for the same number of streams.

## Scalar

These methods were my implementations of Huffman coding split to multiple
streams while sticking to regular scalar code.
This method copies many tricks from Huff0.
Unlike Huff0, my code supports a template-parameterized number of streams.

## AVX Gather

These methods are straightforward vectorizations of the scalar method, with
reads and writes replaced by gather and scatter instructions respectively.


This is the faster approach for decompression. Similar to "scalar" and Huff0,
decoding tables for up to two symbols are used.

##  AVX Permute

These methods use AVX-512 registers instead of lookup tables for encoding and
decoding.
An array of 256 bytes can be stored in four registers, allowing 64 lookups
to be performed using two `vpermi2b`/ byte permute instructions.
This is the faster approach for compression.


# Conclusion

Using AVX-512 methods with 32 streams, I found compression to be 43-50 % faster and decompression
to be 35-38 % faster when compared to Huff0.
I find this a bit underwhelming,
given that each instruction is processing 8 times the amount of data compared
to the scalar algorithm.


I believe the bottleneck is slow gather and scatter instructions.
On AMD Zen 5, gather and scatter instructions are implemented as "microcoded".
According to [Agner's instruction
tables](https://www.agner.org/optimize/instruction_tables.pdf),
the used gather instruction (`vpgatherqq`/`_mm512_i64_gather_epi64`) 
results in
40 [micro-ops (μops)](https://en.wikipedia.org/wiki/Micro-operation).
Possible future CPUs with faster gather and scatter may see
higher compression and decompression speeds using these AVX-512
methods.

# About this code

This code is experimental, and not suited for production.
In particular, the code makes no serious effort to be safe
against malformed "compressed" data.
I believe with effort it is possible to safeguard against
such data with minimal performance cost.

