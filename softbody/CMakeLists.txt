option(SOFTBODY_ENABLE_CUDA "Enable CUDA backend")
option(SOFTBODY_ENABLE_TRACY "Enable the Tracy profiler")

add_subdirectory(src)
target_include_directories(softbody PUBLIC include/)

