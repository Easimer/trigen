# List of build configuration options
| Identifier             |   Type   | Description                                                                     |
|------------------------|----------|---------------------------------------------------------------------------------|
| `BUILD_LEGACY_STUFF`   |  `BOOL`  | Should old projects like `bark_test`, `ifs_test` be built                       |
| `FBX_SDK_DIR`          |  `BOOL`  | Path to the Autodesk FBX SDK                                                    |
| `FBX_SDK_BUILD_TYPE`   | `STRING` | What kind of FBX SDK lib to link against (debug, release)                       |
| `SOFTBODY_CLANG_TIDY`  |  `BOOL`  | Execute clang-tidy on the code base                                             |
| `SOFTBODY_ENABLE_CUDA` |  `BOOL`  | Enable CUDA backend in the softbody library                                     |
| `SOFTBODY_TESTBED_QT`  |  `BOOL`  | Build the Qt testbed                                                            |

# Building

Acquire these:
- A C++17 compiler
- CMake
- SDL2 development libraries
- SDL2_ttf development libraries
- OpenCL development libraries
- Qt5 development libraries
- [Autodesk FBX SDK](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0)
- NVIDIA CUDA Toolkit (if you want to build the CUDA compute backend of the softbody library)

*NOTE: you won't need Qt5 or the FBX SDK unless you turn the CMake option `SOFTBODY_TESTBED_QT` on*

## Windows

On Windows, download and install CMake from [their website](https://cmake.org/download/);
then either build SDL2/SDL2_ttf from source or
download the prebuilt libraries from <https://libsdl.org/download-2.0.php> and
<https://www.libsdl.org/projects/SDL_ttf/> (e.g. if you use MSVC you will need SDL2-devel-2.0.12-VC.zip and  SDL2_ttf-devel-2.0.15-VC.zip).
To install OpenCL you'll need the development package appropriate for your platform:
- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- Intel OpenCL SDK: https://software.seek.intel.com/intel-opencl?os=windows

## Linux
On Linux, download and install CMake, SDL2, SDL2_ttf, OpenCL and Qt5 from your package manager.
If you're building the Qt testbed as well, then you'll need to install the libxml2 and zlib development libraries too.

On Debian-based systems (Debian/Ubuntu) these packages are called `cmake libsdl2-dev libsdl2-ttf-dev ocl-icd-opencl-dev qtbase5-dev libxml2-dev zlib-dev libz-dev`.

On RPM-based systems (RHEL/CentOS/Fedora) these packages are called `cmake SDL2-devel SDL2_ttf-devel SDL2-static ocl-icd-devel qt5-devel libxml2-devel zlib-devel`.

## Generate build files

Now, configure and generate the build scripts using CMake.

If you've installed SDL2 and SDL2_ttf from your package manager, then CMake will automatically find them.
However, if you're building on Windows or your SDL2 binaries are built from source, then you'll need to provide
CMake with a path to these binaries.

On Windows these are:
- `SDL2MAIN_LIBRARY` should point to the `SDL2main.lib` file (`X:/SDL2-2.0.12/lib/x64/SDL2main.lib`)
- `SDL2TTFMAIN_LIBRARY` should point to the library directory (`X:/SDL2_ttf-2.0.15/lib/x64/`)
- `SDL2_DIR` should point to where you've extracted/installed the development libraries (`X:\SDL2-2.0.12`)
- `SDL2_INCLUDE_DIR` should point to the SDL2 header directory (`X:\SDL2-2.0.12\include`)
- `SDL2_LIBRARY` should point to the import library (`X:/SDL2-2.0.12/lib/x64/SDL2.lib`)
- `SDL2TTF_INCLUDE_DIR` should point to the SDL2_ttf header directory (`X:\SDL2_ttf-2.0.15\include`)
- `SDL2TTF_LIBRARY` should point to the SDL2_ttf library directory (`X:\SDL2_ttf-2.0.15\lib\x64`)
- `Qt5_DIR` should point to the directory that contains the `Qt5Config.cmake` file (`X:\Qt\Qt5.12.9\5.12.9\msvc2017_64\lib\cmake\Qt5`)
- `FBX_SDK_DIR` should point to the directory where you've installed the FBX SDK; it contains directories like `include` and `lib` (`X:\Autodesk\FBX\FBX SDK\2020.1.1`)
On Windows CMake will probably only ask for `SDL2_DIR`, `SDL2TTF_LIBRARY`, `SDL2TTF_INCLUDE_DIR`, `FBX_SDK_DIR` and `Qt5_DIR`, as the rest of them will be inferred from these three arguments.

## CUDA
To enable the CUDA compute backend in the softbody library, set the configuration option `SOFTBODY_ENABLE_CUDA` to `ON`.

CMake may ask for the path to nvcc (`CMAKE_CUDA_COMPILER`).
Now if CMake fails to configure the project and tells you that it still can't find the compiler (despite telling it the exact location) that means that the GCC compiler on your platform is not yet supported by NVIDIA.

You can check whether this is the case by trying to manually compile a CUDA program. 

If nvcc says that your GCC is unsupported, then you'll need an older version (for CUDA 11.0 it's GCC 8), then set `CMAKE_CUDA_FLAGS` to `-ccbin cuda-g++`.

## Example CMake invocation on Fedora 32

`cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_FLAGS="-ccbin cuda-g++" -DSOFTBODY_TESTBED_QT=ON -DSOFTBODY_ENABLE_CUDA=ON -DSOFTBODY_CLANG_TIDY=ON -DFBX_SDK_DIR=/fbxsdk/ -GNinja /trigen/`

# Running
Right now the build system doesn't copy all the files to the build directory that are needed to run the programs and you must do this manually.
- Copy all files from /shaders/ to the working directory
- Copy all files from /fonts/ to the working directory
- (Windows only) Copy your SDL2 and SDL2_ttf DLLs to the working directory