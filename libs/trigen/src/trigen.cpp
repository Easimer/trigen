// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "trigen.h"
#include "trigen.hpp"

extern "C" {

Trigen_Status TRIGEN_API Trigen_CreateSession(Trigen_Session *session, Trigen_Parameters const *params) {
    *session = nullptr;
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_DestroySession(Trigen_Session session) {
    return Trigen_OK;
}

}
