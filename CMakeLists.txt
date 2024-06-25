cmake_minimum_required(VERSION 3.10)

project(lidarsim C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

add_executable(basic examples/basic.c)

# Enable optimization flags based on the compiler
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # For GCC and Clang
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -flto -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -fomit-frame-pointer")
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
    # For Intel Compiler
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -xHost -ipo -no-prec-div")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -xHost -ipo -no-prec-div")
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    # For MSVC
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /fp:fast /GL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /fp:fast /GL")
endif()

# Enable link-time optimization (LTO) for GCC and Clang
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
endif()

target_link_libraries(basic m)
