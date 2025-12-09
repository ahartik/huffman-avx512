#!/usr/bin/env python

import sys
import re
import json

if len(sys.argv) < 3:
    print("Usage: {} benchmark_output.json BM_BenchmarkName".format(sys.argv[0]))
    sys.exit(1)

fpath = sys.argv[1]
bm_name = sys.argv[2]

with open(fpath) as f:
    data = json.load(f)

def get_template(s):
    return s[s.find('<')+1:-1]

def get_name_and_streams(tstr):
    if tstr.startswith("::huffman::HuffmanCompressorMulti"):
        return "Scalar", get_template(tstr)
    elif tstr.startswith("::huffman::HuffmanCompressorAvxGather"):
        return "AVX-512 Gather", get_template(tstr)
    elif tstr.startswith("::huffman::HuffmanCompressorAvxPermute"):
        return "AVX-512 Permute", get_template(tstr)
    elif tstr == "::huffman::Huff0Compressor":
        return "Huff0","4"
    else:
        # print("Weird tstr: ''".format(tstr))
        return None, None

bps = []
compress = dict()
decompress = dict()

for bm in data["benchmarks"]:
    name = bm['name']
    # print("name='{}'".format(name))
    if name.find(bm_name) >= 0:
        tname,streams = get_name_and_streams(get_template(name))
        if tname != None:
            key = "{}/{}".format(tname, streams)
            bps = bm['bytes_per_second']
            if name.startswith("BM_Compress"):
                compress[key] = bps
            elif name.startswith("BM_Decompress"):
                decompress[key] = bps
            else:
                print("Only benchmarks starting with BM_Compress/BM_Decompress are supported, "+
                      "got '{}'".format(name))
                sys.exit(1)


print(compress)
print(decompress)

def bps_str(x):
    return "{} MiB/s".format(int(x / (2**20)))

# AVX-512 Gather | 16 | 1943 MiB/s | 3419 MiB/s
print("Method | Streams | Compress | Decompress");
print("-------|--- | ---| ---");
for key in compress:
    tname,streams = key.split('/')
    print("{} | {} | {} | {}".format(
        tname, streams, bps_str(compress[key]), bps_str(decompress[key])))
