// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: constexpr map
//

#pragma once

#include <utility>
#include <array>

template<typename K, typename V, size_t Size>
struct Constexpr_Map {
    using Data_Source = std::array<std::pair<K, V>, Size>;
    Data_Source data;

    constexpr V at(K const& key) const {
        auto const it = std::find_if(begin(data), end(data), [&key](auto const& kv) { return kv.first == key; });

        if (it != end(data)) {
            return it->second;
        } else {
            throw std::range_error("Key was not found in map!");
        }
    }
};

template<typename K, typename V, size_t Size>
struct Bijective_Constexpr_Map {
    using Data_Source = std::array<std::pair<K, V>, Size>;
    Data_Source data;

    constexpr V at(K const& key) const {
        auto const it = std::find_if(begin(data), end(data), [&key](auto const& kv) { return kv.first == key; });

        if (it != end(data)) {
            return it->second;
        } else {
            throw std::range_error("Key was not found in map!");
        }
    }

    constexpr K at(V const& value) const {
        auto const it = std::find_if(begin(data), end(data), [&value](auto const& kv) { return kv.second == value; });

        if (it != end(data)) {
            return it->first;
        } else {
            throw std::range_error("Key was not found in map!");
        }
    }
};
