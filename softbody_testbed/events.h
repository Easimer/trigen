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

#include <SDL.h>

class IEvent_Handler {
public:
    virtual bool on_event(SDL_Event const& ev, float delta) = 0;

    class Event_Handler_Caller {
    public:
        bool operator()(IEvent_Handler* h, SDL_Event const& ev, float delta) {
            return h->on_event(ev, delta);
        }
    };

    using caller_t = Event_Handler_Caller;
};

