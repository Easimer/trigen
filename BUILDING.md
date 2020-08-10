# Building

Acquire these:
- A C++17 compiler
- CMake
- SDL2 development libraries
- SDL2_ttf development libraries
- OpenCL development libraries
- Qt5 development libraries

*NOTE: you won't need Qt5 unless you turn the CMake option `SOFTBODY_TESTBED_QT` on*

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

On Debian-based systems (Debian/Ubuntu) these packages are called `cmake libsdl2-dev libsdl2-ttf-dev ocl-icd-opencl-dev qtbase5-dev`.

On RPM-based systems (RHEL/CentOS/Fedora) these packages are called `cmake SDL2-devel SDL2_ttf-devel SDL2-static ocl-icd-devel qt5-devel`.

## Generate build files

Now, configure and generate the build scripts using CMake. You can build either out-of-tree or in-tree. 

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
On Windows CMake will probably only ask for `SDL2_DIR`, `SDL2TTF_LIBRARY`, `SDL2TTF_INCLUDE_DIR` and `Qt5_DIR`, as the rest of them will be inferred from these three arguments.

# Running
Right now the build system doesn't copy all the files to the build directory that are needed to run the programs and you must do this manually.
- Copy all files from /shaders/ to the working directory
- Copy all files from /fonts/ to the working directory
- (Windows only) Copy your SDL2 and SDL2_ttf DLLs to the working directory