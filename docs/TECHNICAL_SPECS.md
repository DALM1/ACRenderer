Spécification technique — Moteur graphique en Objective‑C++ (.mm)

Objectif : concevoir un moteur graphique multiplateforme moderne dont tout le code source est écrit en Objective‑C++ (.mm), tirant parti du meilleur de C, C++ et Objective‑C (runtime dynamique) : coeur (moteur) en C++ pour performances, API et scripting exposés en ObjC, et possibilité d’utiliser directement du C pour les parties système.

⸻

Table des matières
	1.	Vision & objectifs
	2.	Contraintes & cibles plateformes
	3.	Architecture globale
	4.	Modules et responsabilités
	5.	Format de fichiers et layout du repo
	6.	Toolchain, build et CI
	7.	Gestion des ressources (assets)
	8.	Pipeline de rendu (abstraction)
	9.	Scene graph / ECS / entités
	10.	Système de mémoire & threading
	11.	API Objective‑C pour l’intégration et tooling
	12.	Interop C/C++/ObjC — règles & bonnes pratiques
	13.	Shaders, material system, et pipeline GPU
	14.	Input, audio, réseau et I/O
	15.	Outils développeur : editor, profiler, hot‑reload
	16.	Tests, QA et perf targets
	17.	Roadmap & jalons
	18.	Annexes: exemples de CMake, conventions de code, checklist de portage

⸻

1. Vision & objectifs
	•	But : fournir un moteur performant et flexible, entièrement écrit en .mm pour unifier toolchain et permettre :
	•	mélange sans friction C/C++/ObjC
	•	export facile d’API Objective‑C pour des frontends Cocoa/GNUstep/ObjFW
	•	interop simple avec bibliothèques C/C++ existantes (SDL, Vulkan, OpenGL, bgfx, stb)
	•	Cas d’usage : jeux 2D/3D, visualisateurs, prototypes interactifs, outils de rendu temps réel.

2. Contraintes & cibles plateformes
	•	Cibles : Linux (glibc), Windows (Clang), macOS (Xcode optional), potentiellement BSD.
	•	Compilateur requis : Clang (LLVM) — support Objective‑C++ moderne et ARC si souhaité.
	•	Runtime ObjC : libobjc2 (Linux/Windows) + GNUstep / ObjFW pour couches Foundation si besoin.
	•	Backends graphiques : Vulkan + Metal (macOS) + D3D12 (Windows) + fallback OpenGL.
	•	Performance : latence <16ms @60FPS pour scenes moyennes, budget par frame modulable.

3. Architecture globale
	•	Couche bas‑niveau (C/C++) : drivers, allocateurs, abstraction plateforme, math, containers performants.
	•	Coque moteur (C++ & ObjC++) : Renderer, ResourceManager, Scene, EntitySystem, JobSystem.
	•	API publique (Objective‑C) : façades ergonomiques (classes MMRenderer, MMScene, MMAssetManager) pour usage par apps et UIs.
	•	Scripting & extension : bindings (Lua / JavaScript) exposés via C API et adaptateurs ObjC.
	•	Editor / Tools : applicatif en ObjC (Cocoa/GNUstep) ou front web, communiquant via sockets/IPC.

4. Modules et responsabilités
	•	Platform : fenêtrage, input, clipboard, filewatch, dynamic library loader.
	•	RendererCore : abstraction backend, swapchain, command buffers, resource binding.
	•	ResourceManager : loader asynchrone, caching, dedup, streaming.
	•	Scene/ECS : entités, composants (Transform, Mesh, Material, Light, Camera), systèmes.
	•	Math : vecteurs, matrices, quaternions, SIMD optimisé (optionnel).
	•	JobSystem : pool threads + task graph (futures/promise léger en C++).
	•	Memory : custom allocators (stack, linear, pool), debug alloc, leak detection.
	•	Audio : abstraction (OpenAL / ALSA / WASAPI / CoreAudio).
	•	Network : sockets non‑bloquants, RPC simple pour editor remote.
	•	Debug : overlays, GPU capture hooks, instrumentation.

5. Format de fichiers et layout du repo

/ (root)
  /engine
    /src        # .mm, .h, .cpp, .c
    /include    # headers publiques (.h / .hpp / .hobjc)
  /platform
    /linux
    /windows
    /macos
  /tools
    editor/    # app ObjC ou Qt
    pipeline/  # asset conversion
  /third_party
  /examples
  CMakeLists.txt
  LICENSE
  README.md

	•	Conventions : tous les fichiers compilés C/C++/ObjC++ utilisent extension .mm. Headers C++ .hpp ou .h++, Headers C/ObjC .h.

6. Toolchain, build et CI
	•	Local dev : Clang + LLVM, CMake (>=3.20), ninja
	•	Dependencies : Vulkan SDK, SDL2 (optional), libobjc2, GNUstep (dev runtime), glm (optionnel), stb, tinyobjloader, bgfx (optionnel)
	•	Flags recommandés : -std=c++20, -fobjc-arc (optionnel), -fobjc-runtime=gnustep-2.0 (pour Linux), sanitizer flags en debug.
	•	CI : GitHub Actions matrix (linux-clang, windows-clang, macos-xcode), tests unitaires, build examples, smoke tests.

7. Gestion des ressources (assets)
	•	Pipeline : sources (FBX/GLTF, PNG, WAV, OBJ) → conversion → runtime binary package (.pak / custom) avec manifest.
	•	Resource types : Mesh, Material, Texture, Shader, Animation, Audio.
	•	Streaming : loader asynchrone + LRU cache, priorité based streaming.
	•	Hot‑reload : filewatcher → reimport minimal → swap resources à chaud.

8. Pipeline de rendu (abstraction)
	•	Design : Renderer API agnostique (interface IRendererBackend) et backends spécifiques (Vulkan, Metal, D3D12, GL).
	•	Command model : enregistrement de command buffers via builder pattern, submission au GPU.
	•	Resource binding : descriptor sets / root signatures abstraits.
	•	Frame graph : optionnel mais recommandé — définir passes, dépendances, transitions.

9. Scene graph / ECS / entités
	•	Choix : ECS hybride (entités id + composants structs) pour perf, + scene graph pour hiérarchies transform.
	•	Composants core : Transform, Renderable (mesh + material), Camera, Light, PhysicsBody, Script.
	•	Systèmes : RenderSystem, CullingSystem (frustum + occlusion optional), PhysicsSystem (third party), AnimationSystem.

10. Système de mémoire & threading
	•	Allocateurs : ephemeral stack per frame, pool allocators pour meshes/textures, arena pour resources.
	•	Threading model : main thread (platform + render submit), render worker(s), job workers.
	•	Synchronisation : minimise locks — use lockless queues, atomics pour counters, fences/semaphores GPU.

11. API Objective‑C pour l’intégration et tooling
	•	Objectifs API : simple, idiomatique ObjC, classes immutables pour config, delegates pour callbacks.
	•	Exemples :
	•	@interface MMRenderer : NSObject … méthodes start, stop, renderFrame, setBackend:
	•	@protocol MMResourceDelegate pour hook import/progress.
	•	Binding pattern : C++ core encapsulé dans struct Impl; pointeur opaque détenu par l’objet ObjC.

12. Interop C/C++/ObjC — règles & bonnes pratiques
	•	Headers : séparer interface C compatible (extern “C”) vs headers C++/ObjC. Inclure C++ dans .mm seulement.
	•	Ownership : clairement documenter ownership (ARC managed vs unique_ptr/shared_ptr). Utiliser objc_retain/objc_release via ARC rules.
	•	Exceptions : éviter de traverser frontières — ne pas lancer exceptions C++ dans ObjC @catch et vice versa.

13. Shaders, material system, et pipeline GPU
	•	Shaders : authoring HLSL/GLSL/Metal; use cross‑compiler (glslang, SPIRV‑cross, DXC) → deliver SPIR‑V / Metal binary.
	•	Material : graph basé sur nodes, with parameter blocks, texture slots, and render states.
	•	PBR : support standard PBR (albedo, metallic, roughness, normal, AO).

14. Input, audio, réseau et I/O
	•	Input : abstraction layer mapping SDL/WinAPI/Quartz events → unified events.
	•	Audio : streaming + 3D audio; sample mixer, buffer queuing, low latency.
	•	Réseau : UDP/TCP wrappers, simple client/server demo pour multiplayer/editor sync.

15. Outils développeur : editor, profiler, hot‑reload
	•	Editor : app native ObjC (Cocoa/GNUstep) ou multi‑tool (Electron/Qt) communiquant avec engine.
	•	Profiler : instrumentation (CPU scopes), GPU timings (timestamp queries), memory snapshots.
	•	Debug GUI : ImGui integrable (C++), exposé via Objective‑C wrapper pour util GUI si souhaité.

16. Tests, QA et perf targets
	•	Tests unitaires : math, resource loader, small renderer tests (offscreen render compare)
	•	Smoke tests : build + run examples on CI every commit
	•	Perf targets : maintain stable 60FPS on target hardware; measure drawcall budget, memory use.

17. Roadmap & jalons (exemple)
	1.	Prototype minimal (MVP) — affichage d’un triangle (Vulkan/GL) — 2 semaines
	2.	Resource loader + texture sampler — 2 semaines
	3.	Mesh rendering + material PBR — 4 semaines
	4.	Scene graph + camera + culling — 3 semaines
	5.	Editor simple + hot reload — 4 semaines
	6.	Backends multiplateformes + CI — 3 semaines

18. Annexes

Exemple CMake minimal (squelette)

cmake_minimum_required(VERSION 3.20)
project(MMEngine LANGUAGES C CXX ObjCXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Vulkan REQUIRED)
# find libobjc2/GNUstep on Linux
add_subdirectory(engine)
add_executable(app examples/main.mm)
target_link_libraries(app PRIVATE MMEngine ${Vulkan_LIBRARIES} objc gnustep-base)

Conventions de code
	•	Fichiers source .mm
	•	Headers exposées .h (C/ObjC), .hpp ou .hh (C++)
	•	Préfixe types ObjC MM pour engine (ex: MMRenderer)

⸻

Prochaines étapes proposées
	•	Choisir le backend graphique principal (Vulkan recommandé) et valider le périmètre initial.
	•	Générer un template CMake + example triangle.mm pour tests multiplateformes.
	•	Définir la liste minimale d’APIs ObjC publiques pour la v0 (start/stop, loadMesh, createMaterial, attachEntity).

⸻

Si tu veux, je peux :
	•	Générer le CMake complet + arborescence de projet (squelettes de fichiers .mm) ;
	•	Ou écrire le main triangle (file triangle.mm) + Makefile/CMake prêt à compiler sous Linux/Clang ;
	•	Ou détailler le design du ResourceManager ou du FrameGraph.

Dis‑moi quelle tâche tu veux attaquer en premier.
