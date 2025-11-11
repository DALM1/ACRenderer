<h1 align="center">
  <br>
    ACRenderer
  <br>
  <img src="https://github.com/DALM1/ACRenderer/blob/main/ACRender_default_screen.png?raw=true" alt="ACRenderer" width="800">
  <br>
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExNGo4b21xZnkyaHhkOWQ3b2M5ampnZ2QwZ3RvcjhweWJub3ZmcXE3ayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Z3GuJnqqAKuXGwA7nI/giphy.gif" width="600">
</h1>

Simple Metal-based renderer with a minimal glTF 2.0 (GLB) loader and an orbit camera example.

Project Structure
- `engine/` — Core rendering engine (`MMRenderer`) with headers and sources.
- `examples/` — macOS example app (`ACRenderer`) using the engine.
- `assets/` — Models and textures for examples.
- `docs/` — Technical notes and specifications.

Build & Run (macOS)
- Requirements: Xcode toolchain, CMake ≥ 3.20.
- Configure: `cmake -S . -B build`
- Build: `cmake --build build -j4`
- Run: `./build/ACRenderer`

Controls
- Orbit: click-drag to rotate.
- Zoom: mouse wheel.
- Reset: press `r`.

Notes
- The example loads `assets/models/leda_va_museum.glb`.
- GLB loader supports one primitive with POSITION, TEXCOORD_0, and uint16/uint32 indices; embedded JPEG textures via bufferView.# ACRenderer
