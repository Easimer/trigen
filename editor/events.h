// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: event handling utilities
//

#pragma once

#include <type_traits>
#include <vector>

template<class TEvent, class THandler>
class Chain_Of_Responsibility {
public:
    template<class ...Args>
    void attach(THandler h, Args... args) {
        attach(h);
        attach(args...);
    }

    void attach(THandler h) {
        handlers.push_back(h);
    }

    using THandler_NotPtr = typename std::remove_pointer<THandler>::type;
    using caller_t = typename THandler_NotPtr::caller_t;

    template<typename ...T>
    bool handle(TEvent const& ev, T... h) {
        caller_t caller;
        for (auto& handler : handlers) {
            if (caller(handler, ev, h...)) {
                return true;
            }
        }
        return false;
    }
private:
    std::vector<THandler> handlers;
};
