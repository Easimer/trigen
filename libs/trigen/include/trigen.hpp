// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: C++ helpers for libtrigen
//

#pragma once

#ifdef __cplusplus

#include <trigen.h>

#include <optional>
#include <exception>

namespace trigen {

class Exception : public std::exception {
public:
    Exception(Trigen_Status rc)
        : rc(rc) {
    }

    Trigen_Status code() const noexcept {
        return rc;
    }

private:
    Trigen_Status rc;
};

class Session {
public:
    static Session make(Trigen_Parameters const &parameters) {
        Trigen_Session handle = nullptr;
        Trigen_Status rc;
        if ((rc = Trigen_CreateSession(&handle, &parameters)) != Trigen_OK) {
            throw Exception(rc);
        }

        return Session(handle);
    }

    operator Trigen_Session() const noexcept {
        return _handle;
    }

    Trigen_Session handle() const noexcept {
        return _handle;
    }

    void grow(float time) {
        Trigen_Status rc;

        if ((rc = Trigen_Grow(_handle, time)) != Trigen_OK) {
            throw Exception(rc);
        }
    }
    
    ~Session() {
        if (_handle != nullptr) {
            Trigen_DestroySession(_handle);
        }
    }

    Session()
        : _handle(nullptr) {
    }
    Session(Session const &) = delete;
    void operator=(Session const &) = delete;

    Session(Session &&other) noexcept : _handle(nullptr) {
        std::swap(_handle, other._handle);
    }

    Session &operator=(Session &&other) noexcept {
        if (_handle != nullptr) {
            Trigen_DestroySession(_handle);
            _handle = nullptr;
        }
        std::swap(_handle, other._handle);
    }

protected:
    Session(Trigen_Session handle)
        : _handle(handle) {
    }

private:
    Trigen_Session _handle;
};

class Collider {
public:
    static Collider make(Session &session, Trigen_Collider_Mesh const &mesh, Trigen_Transform const &transform) {
        Trigen_Collider handle = nullptr;
        Trigen_Status rc;
        if ((rc = Trigen_CreateCollider(&handle, session, &mesh, &transform)) != Trigen_OK) {
            throw Exception(rc);
        }

        return Collider(handle);
    }

    void update(Trigen_Transform const &transform) {
        Trigen_Status rc;

        if ((rc = Trigen_UpdateCollider(_handle, &transform)) != Trigen_OK) {
            throw Exception(rc);
        }
    }

protected:
    Collider(Trigen_Collider handle)
        : _handle(handle) {
    }

private:
    Trigen_Collider _handle;
};

}

#endif /* __cplusplus */
