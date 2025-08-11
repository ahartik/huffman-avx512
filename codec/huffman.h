#pragma once

#include <array>
#include <string_view>
#include <string>

using ByteCounts = std::array<int, 256>;

namespace huffman {
namespace internal {
}

std::string compress(std::string_view raw);
std::string decompress(std::string_view compressed);

}
