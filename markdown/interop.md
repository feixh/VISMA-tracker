## [OpenGL Interoperability with CUDA](https://www.3dgep.com/opengl-interoperability-with-cuda/)

- [PBO](https://www.khronos.org/opengl/wiki/Pixel_Buffer_Object): A Buffer Object used for asynchronous *pixel transfer operations* is called Pixel Buffer Object.
  - PBO are not connected to textures, only used for pixel transfers. The storage remains with the texture.
  - Pixel transfer operation is the act of taking pixel data from an unformatted memory buffer and copying it in OpenGL-owned storage governed by an image format.  Or vice-versa: copying pixel data from image fromat-based storage to unformatted memory.
- [glBufferData](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBufferData.xhtml)
  - Note the last argument `GLenum usage`: a hint to the GL as to *how a buffer object's data store will be accessed*. usage can be broken down into two parts:
    1. the frequency of access: STREAM, STATIC, DYNAMIC
    2. the nature of that access:
      - DRAW: the contents are modified by the application, and used as **the source** for GL drawing and image specification commands.
      - READ: the cotents are modified by reading data from the GL, and used to return that data when queried by the application.
      - COPY: the contents are modified by reading data from the GL, and used as the source for GL drawing and image specification commands.

- Why do we need render buffer for depth component? How is it different from using texture_2d?
    - [Renderbuffer Objects](https://www.khronos.org/opengl/wiki/Renderbuffer_Object) are OpenGL Objects that contain images. They are created and used specifically with Framebuffer Objects. They are optimized for use as render targets, while Textures may not be, and **are the logical choice when you do NOT need to sample (i.e. in a post-pass shader) from the produced image.** If you need to resample (such as when reading depth back in a second shader pass), use Textures instead. Renderbuffer objects also natively accommodate Multisampling (MSAA).

- Register a **Texture Resource** with CUDA
    - Before a texture or buffer can be used by a CUDA application, the buffer (or texture) must be registered.
    - To register an OpenGL texture or render-buffer resource with CUDA, you must use the `cudaGraphicsGLRegisterImage` method. This method will accept an OpenGL texture or render-buffer resource ID as a parameter and provide a pointer to a `cudaGraphicsResource_t` object in return.
    - The `cudaGraphicsResource_t` object is then used to map the memory location defined by the texture object so that it can be used as a texture reference in CUDA later.
- Register a Vertex Buffer or Pixel Buffer with CUDA
  -  Pixel buffers and vertex buffers are more like C-arrays that reside in device memory instead of system memory.
  - To register a buffer object, you need to use the `cudaGraphicsGLRegisterBuffer` method instead.
- Immediately after drawing contents to the framebuffer and unbinding the framebuffer, we can use CUDA to perform post-process steps, such as applying a filter to the image.
- To access the registered resources in CUDA, we must map the resources, during which, the resouces will be locked and any access to them will raise an error. After CUDA computing, we need to un-map (release) the resources.
  - To map a resource to be used in CUDA, use `cudaGraphicsMapResources` method.
- After mapping the resources to CUDA, we need to get a pointer to the device memory that can be used in the CUDA kernel. We need to map the pointer to device memory based on the type of the resources being mapped.
  - If the resource is a texture or render-buffer, use `cudaGraphicsSubResourceGetMappedArray` which map the texture resource to a 2D CUDA array object.
    - However, CUDA array cannot be used directly in a kernel and requires an **additional step(!!!)** to access it, which depends on *how the memory should be used in the kernel*.
      - If the resource will be used as read-only texture in the kernel, then the cuda array must be bound to a *texture reference* that is used within the kernel to access the data.
      - If you need to write to the resource from within the kernel, then you will need to bind the cuda array to a *surface reference* that can be both *read-from and written-to* in the CUDA kernel.
      - *Bind a CUDA array to a texture reference*: use `cudaBindTextureToArray` method. Note we need to **declare the texture reference first** (in the global scope of our CUDA source file? or suitable scopes).
  - If the resource is a vertex buffer or pixel buffer object, use `cudaGraphicsResourceGetMappedPointer` to get a direct pointer to the device memory that refers to the graphics resource.
- Now that we have bound a CUDA array to a texture reference, we can read data from the texture reference and operate on the CUDA array as usual.
- After we finish our operations on the CUDA array, we want to copy data from CUDA array back to the texture which will be shown on the screen. To do that, we can use `cudaMemcpyToArray` method to copy the global memory **to** a CUDA array object which was previously mapped to a texture using `cudaGraphicsSubResourceGetMappedArray`.
- The author of this article nicely provided the [source code](https://drive.google.com/file/d/0B0ND0J8HHfaXT0p1N3ZkSW5kTVU/edit).

## Thread Hierarchy:
  - from most fine grained to coarse level: thread -> block -> grid
- Kernel is an analogue to functions in C/C++
  - Defined with `__global__`
  - Called with *executation configuration* syntax `<<< >>>`
- Thread can be accessed via `threadIdx`, which are essentially 3-vector (though they can be used as 1-/2-vector) to access data in the form vector/matrix/volume.
- type `dim3` to define dimension of blocks/grids.
- Dimension of a block can be accessed inside a function via `blockDim.x/.y/.z` and block index via `blockIdx.x/.y/.z`.

## Memory Hierarchy
- Each thread block has shared memory visible to all the threads of the block.
- Two additional read-only memory spaces accessible by all threads:
  - Constant and texture memory spaces.
  - Since they are accessible by all threads, they are **global**.
  - And they are optimized for different memory usages.
- So each thread carries its own memory space, sees the shared memory space of the enclosing block and also sees the global constant and texture memory spaces.


## References
- [CUDA Memory Model](https://www.3dgep.com/cuda-memory-model/)
- [Slides from NVIDIA](http://www.nvidia.com/content/gtc/documents/1055_gtc09.pdf)
- [OpenGL Interoperability with CUDA](https://www.3dgep.com/opengl-interoperability-with-cuda/)
- CUDA-OpenGL Interoperability is also discussed in the **CUDA C Programming Guide**
- For introduction to CUDA C Programming, see the NVIDIA slides titled **CUDA C/C++ Basics**.


---

## Build System
- See [this article](https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake/) about how to build heterogeneous program with cmake. 
