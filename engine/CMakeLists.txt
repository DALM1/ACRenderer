cmake_minimum_required(VERSION 3.20)

add_library(MMEngine STATIC
    src/MMRenderer.mm
)

target_include_directories(MMEngine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link Apple frameworks used by the engine implementation
if(APPLE)
    target_link_libraries(MMEngine PUBLIC
        "-framework Foundation"
        "-framework QuartzCore"
        "-framework Metal"
        "-framework MetalKit"
    )
endif()