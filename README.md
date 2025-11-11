<h1 align="center">
  <br>
    ACRenderer
  <br>
  <img src="https://github.com/DALM1/SKRender/blob/main/SKRender-demo.png?raw=true" alt="SKRender" width="800">
</h1>
=========

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
