## Tips on OpenGL

### Read values from depth buffer
```C++
Mat depthImg(HEIGHT, WIDTH, CV_32FC1);
glReadPixels(0, 0, WIDTH, HEIGHT, GL_DEPTH_COMPONENT, GL_FLOAT, depthImg.data);
```
see this [wiki page](https://www.khronos.org/opengl/wiki/Pixel_Transfer#Pixel_format).

### Performance and debugging tips on stencil buffer
https://en.wikibooks.org/wiki/OpenGL_Programming/Stencil_buffer#Performances


### SWIG interface
- handle std::string, use ```%include std_string.i```
- handle c/c++ pointers, use ```%include pointer.i```
- How to use SWIG with CMake:
http://www.swig.org/Doc1.3/Introduction.html

## References
- [OpenGL tutorial](https://learnopengl.com/#!Getting-started/Hello-Window)
- [GLFW](http://www.glfw.org/docs/latest/window_guide.html)

- [Silhouette Rendering -- Dragon](http://prideout.net/blog/?tag=opengl-silhouette#downloads)

- [Silhouette Rendering - Kitti](http://sunandblackcat.com/tipFullView.php?l=eng&topicid=15&topic=Cel-Shading)

- [Silhouette Rendering - Hardware](http://www.marctenbosch.com/npr_edges/)

- [Normal Mapping](http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/)

- [Geometry Shader](https://learnopengl.com/#!Advanced-OpenGL/Geometry-Shader) and [this](http://www.informit.com/articles/article.aspx?p=2120983&seqNum=2)

- [Use depth texture for fragment shader](https://stackoverflow.com/questions/23362076/opengl-how-to-access-depth-buffer-values-or-gl-fragcoord-z-vs-rendering-d)

- [How to read values from shader]( https://www.opengl.org/discussion_boards/showthread.php/164903-reading-data-back-from-a-fragment-shader) -- short answer:
write to a framebuffer as texture and read the texture.

- [Common Mistakes](https://www.khronos.org/opengl/wiki/Common_Mistakes)
