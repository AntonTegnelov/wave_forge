## wfc-gpu package

> Planned enhancements for the GPU accelerated implementation.

- [x] **Default fallback shaders**: Create fallback shaders that work even on systems with limited GPU capabilities.
- [x] **Error reporting improvements**: Add detailed and helpful error messages for GPU-specific failures.
- [x] **Multi-dimensional grid support**: Extend the implementation to handle 1D, 2D, and 3D grids seamlessly.
- [x] **Custom entropy heuristics**: Allow custom entropy calculation strategies beyond Shannon entropy.
- [ ] **Performance monitoring**: Add performance tracking capabilities to benchmark and optimize the GPU implementation.
- [ ] **Dynamic rule switching**: Enable changing adjacency rules mid-generation for advanced applications.
- [ ] **Memory optimization**: Reduce memory usage through better buffer management.
- [ ] **Support for larger tile sets**: Optimize to handle very large tile sets efficiently.
- [ ] **Visualization tools**: Built-in tools to visualize the generation progress directly from GPU data.
- [ ] **Multiple backend support**: Abstract the GPU interface to potentially support backends other than wgpu.

### Implementation notes

- Shannon entropy can now be replaced with simpler heuristics like direct counts or weighted counts
- The GPU implementation fully supports the EntropyHeuristicType enum defined in wfc-core
- The different heuristics are implemented directly in the entropy shader for optimal performance

## Architecture

- [x] **Separation of GPU synchronization and algorithm logic**: Refactor to better separate these concerns.
- [x] **Dynamic buffer management**: Implement resizable buffers based on runtime requirements.
- [x] **Proper resource cleanup**: Ensure all GPU resources are properly released using RAII patterns.
- [x] **Abstract hardware specifics**: Create abstraction layers to handle different GPU backends/capabilities.
- [ ] **Reduce Arc nesting**: Simplify ownership model where excessive Arc wrapping occurs.
- [ ] **Modularize shader code**: Split large shader functions into more manageable, testable pieces.
