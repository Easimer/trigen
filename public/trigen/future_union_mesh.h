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
struct Future_Union_Mesh {
    using FM = std::future<Mesh_Builder::Optimized_Mesh>;
    using OT = std::unique_ptr<Future_Union_Mesh>;
    using OFM = std::optional<FM>;
    using V = std::variant <std::monostate, FM, OT>;

    V lhs;
    OT rhs;


    Mesh_Builder::Optimized_Mesh operator()(FM& x) {
        return x.get();
    }

    Mesh_Builder::Optimized_Mesh operator()(OT& x) {
        if (x != NULL) {
            return (Mesh_Builder::Optimized_Mesh)*x;
        } else {
            return {};
        }
    }

    Mesh_Builder::Optimized_Mesh operator()(std::monostate const&) {
        return {};
    }

    explicit operator Mesh_Builder::Optimized_Mesh() {
        auto const x = std::visit(*this, lhs);
        if (rhs != NULL) {
            return x + (Mesh_Builder::Optimized_Mesh)*rhs;
        } else {
            return x;
        }
    }
};

// =============================================
// Purpose: create the union of two FUMs, moving the right-hand side inside the
// left-hand side. 
// =============================================
inline void Union(Future_Union_Mesh& lhs, Future_Union_Mesh&& rhs) {
    auto x = std::make_unique<Future_Union_Mesh>(std::move(lhs));
    auto y = std::make_unique<Future_Union_Mesh>(std::move(rhs));
    lhs = Future_Union_Mesh { std::move(x), std::move(y) };
}

// =============================================
// Purpose: create the union of a FUM and a future Optimized_Mesh.
// =============================================
inline void Union(Future_Union_Mesh& lhs, Future_Union_Mesh::FM&& fm) {
    auto y = std::make_unique<Future_Union_Mesh>(std::move(lhs));
    lhs = Future_Union_Mesh{ std::move(fm), std::move(y) };
}
