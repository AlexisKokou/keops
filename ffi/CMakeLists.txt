cmake_minimum_required(VERSION 3.15)

project(x_square_ffi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

execute_process(
  COMMAND python3 -c "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE JAXLIB_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Using JAXLIB_INCLUDE = ${JAXLIB_INCLUDE}")
include_directories(${JAXLIB_INCLUDE})

add_library(x_square SHARED x_square.cpp)

install(TARGETS x_square
        LIBRARY DESTINATION .
        ARCHIVE DESTINATION .
        RUNTIME DESTINATION .)