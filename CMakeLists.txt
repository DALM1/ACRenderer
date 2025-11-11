cmake_minimum_required(VERSION 3.20)
project(ACRenderer LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(engine)

add_executable(ACRenderer examples/main.mm)
target_link_libraries(ACRenderer PRIVATE MMEngine "-framework AppKit")
target_compile_definitions(ACRenderer PRIVATE ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")
