#pragma once

#include <array>
#include <string_view>
#include <string>

using ByteCounts = std::array<int, 256>;

namespace huffman {
namespace internal {
}

std::string Compress(std::string_view raw);
std::string Decompress(std::string_view compressed);

}
