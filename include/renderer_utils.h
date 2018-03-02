//
// Created by visionlab on 2/8/18.
//
// Utility functions for OpenGL based Render Engine

#pragma once
namespace feh {

/// \brief: Check the id of current framebuffer under use
void CheckCurrentFramebufferId();
/// \brief: Print OpenGL and GLSL version information.
void PrintGLVersionInfo();
/// \brief: Create a gaussian kernel
/// \param kernel: Returned kernel values, always use 1 dim vector, but actual kernel can be 2-dim
/// \param kernel_size: Size of the kernel.
/// \param sigma: Standard deviation of gaussian distribution.
void CreateGaussianKernel(std::vector<float> &kernel, int kernel_size, float sigma);
/// \brief: Convert buffered z value in depth buffer to actual depth.
/// \param zb: z in depth buffer
/// \param z_near: near plane distance
/// \param z_far: far plane distance
template <typename T>
T LinearizeDepth(T zb, T z_near, T z_far) {
    return 2 * z_near * z_far /
        (z_far + z_near - (2 * zb - 1) * (z_far - z_near));
}



}
