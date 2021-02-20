// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: FILE* based serializer for `softbody`
//

#include <memory>
#include <softbody.h>

namespace sb {
    std::unique_ptr<sb::IDeserializer> make_deserializer(char const *path);
    std::unique_ptr<sb::ISerializer> make_serializer(char const *path);
}
