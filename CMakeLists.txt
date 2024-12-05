cmake_minimum_required(VERSION 3.19)
project(ada_2024_project_outliars)

set(CMAKE_CXX_STANDARD 20)

# Detect compiler
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Using GCC or Clang compiler")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message(STATUS "Using MSVC compiler")
else()
    message(WARNING "Unsupported compiler detected")
endif()

# Detect OS
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Configuring for Windows")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Configuring for Linux")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Configuring for macOS")
else()
    message(WARNING "Unknown operating system")
endif()

# Set compiler flags based on OS and compiler
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # General flags for GCC/Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -fopenmp")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fstack-protector-strong -fsanitize=undefined,address")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -mtune=native -flto -fuse-linker-plugin -fno-plt -finline-functions -fomit-frame-pointer -fno-math-errno -fno-rtti -frename-registers")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # Flags for MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif()

# Print the flags that are used in the current configuration
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

# Print the mode that is used in the current configuration
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring for Debug mode")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring for Release mode")
else()
    message(WARNING "Unknown build type")
endif()


add_executable(graph_stats src/graph_stats.cxx)
