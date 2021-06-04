// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: common types
//

#pragma once

#include <softbody.h>

class ILogger {
public:
    virtual ~ILogger() = default;


    virtual void log(sb::Debug_Message_Source s, sb::Debug_Message_Type t, sb::Debug_Message_Severity l, char const* fmt, ...) = 0;
};