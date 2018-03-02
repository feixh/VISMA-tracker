- The compute shader is responsible for fetching the input data via
  - Texture access
  - Arbitrary image load
  - Shader storage blocks
  - Other forms of interface
  -
and writing out the results to an image or shader storage block.

- *Work group* -- analogue to threads block in CUDA: 3 dimensional.
- *Compute shader invocation* -- analogue to threads in CUDA: number of invocations (a.k.a. local size, also 3 dimensional) defined by the compute shader itself, NOT by the caller.
- Purpose of the distinction between work groups and invocations: invocations within a work group can communicate through a set of *shared variables and special functions*. Invocations in different work groups (within the same cmpute shader dispatch) cannot effectively communicate.
- To launch compute operations:
  1. use `glBindProgramPileline` or `glUseProgram` to activate the desired compute shader as usual
  2. Dispatch the compute shader via:
```c
void glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
```

- Built-in input variables:
  - in uvec3 gl_NumWorkGroups;
  - in uvec3 gl_WorkGroupID;
  - in uvec3 gl_LocalInvocationID;
  - in uvec3 gl_GlobalInvocationID;
  - in uint  gl_LocalInvocationIndex;

- Local size of a compute shader is defined within the shader using a special layout input declaration:
```GLSL
layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;
```


## References
- [compute shader wiki](https://www.khronos.org/opengl/wiki/Compute_Shader)
- For compute shader IO, see [Shader Storage Buffer Object](https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object)
- Maybe [this article](http://antongerdelan.net/opengl/compute.html)
