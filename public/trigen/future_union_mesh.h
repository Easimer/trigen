// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Future_Union_Mesh
//

#pragma once
#include <future>
#include <memory>
#include <optional>
#include <variant>
#include <trigen/meshbuilder.h>

// =============================================
// Purpose: a union of two or more meshes that are results of an asynchronous
// operation
// =============================================
template<typename Vertex>
struct Future_Union_Mesh {
    using FM = std::future<Optimized_Mesh<Vertex>>;
    using OT = std::unique_ptr<Future_Union_Mesh>;
    using OFM = std::optional<FM>;
    using V = std::variant <std::monostate, FM, OT>;

    V lhs;
    OT rhs;


    Optimized_Mesh<Vertex> operator()(FM& x) {
        return x.get();
    }

    Optimized_Mesh<Vertex> operator()(OT& x) {
        if (x != NULL) {
            return (Optimized_Mesh<Vertex>)*x;
        } else {
            return {};
        }
    }

    Optimized_Mesh<Vertex> operator()(std::monostate const&) {
        return {};
    }

    explicit operator Optimized_Mesh<Vertex>() {
        auto const x = std::visit(*this, lhs);
        if (rhs != NULL) {
            return x + (Optimized_Mesh<Vertex>)*rhs;
        } else {
            return x;
        }
    }
};

// =============================================
// Purpose: create the union of two FUMs, moving the right-hand side inside the
// left-hand side. 
// =============================================
template<typename Vertex>
inline void Union(Future_Union_Mesh<Vertex>& lhs, Future_Union_Mesh<Vertex>&& rhs) {
    auto x = std::make_unique<Future_Union_Mesh<Vertex>>(std::move(lhs));
    auto y = std::make_unique<Future_Union_Mesh<Vertex>>(std::move(rhs));
    lhs = Future_Union_Mesh<Vertex> { std::move(x), std::move(y) };
}

// =============================================
// Purpose: create the union of a FUM and a future Optimized_Mesh.
// =============================================
template<typename Vertex>
inline void Union(Future_Union_Mesh<Vertex>& lhs, std::future<Optimized_Mesh<Vertex>>&& fm) {
    auto y = std::make_unique<Future_Union_Mesh<Vertex>>(std::move(lhs));
    lhs = Future_Union_Mesh<Vertex> { std::move(fm), std::move(y) };
}
