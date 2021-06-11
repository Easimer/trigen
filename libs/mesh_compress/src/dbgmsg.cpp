// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include <cstdio>
#include <cstdarg>

TMC_API
ETMC_Status
TMC_SetDebugMessageCallback(TMC_Context context, TMC_Debug_Message_Proc proc, void* user) {
    if (context == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    context->dbgMsg = proc;
    context->dbgMsgUser = user;

    return k_ETMCStatus_OK;
}

static void TMC_Printf(TMC_Context context, char const* fmt, std::va_list va, TMC_Bool is_error) {
    assert(context);
    assert(fmt);

    if (!context || !fmt || !context->dbgMsg) {
        return;
    }

    char *buf = nullptr;
    int res;
    res = vsnprintf(buf, 0, fmt, va);

    if (res < 0)
        return;

    buf = new char[res + 1];
    res = vsnprintf(buf, res + 1, fmt, va);

    context->dbgMsg(context->dbgMsgUser, buf, is_error ? k_ETMCMsgLevel_Error : k_ETMCMsgLevel_Info);

    delete[] buf;
}

void TMC_Print(TMC_Context context, char const* fmt, ...) {
    std::va_list va;
    va_start(va, fmt);
    TMC_Printf(context, fmt, va, false);
    va_end(va);
}

void TMC_PrintError(TMC_Context context, char const* fmt, ...) {
    std::va_list va;
    va_start(va, fmt);
    TMC_Printf(context, fmt, va, true);
    va_end(va);
}
