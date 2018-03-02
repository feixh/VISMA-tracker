//
// Created by visionlab on 2/8/18.
//
#include "renderer.h"

namespace feh {

void CheckCurrentFramebufferId() {
    GLint drawFboId = 0, readFboId = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFboId);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFboId);
    LOG(INFO) << "before: draw fb id=" << drawFboId << " " << "read fb id=" << readFboId;
}

void PrintGLVersionInfo() {
    const GLubyte *renderer = glGetString(GL_RENDERER);
    const GLubyte *vendor = glGetString(GL_VENDOR);
    const GLubyte *version = glGetString(GL_VERSION);

    const GLubyte *glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION);
    LOG(INFO) << "GL Renderer" << vendor;
    LOG(INFO) << "GL Verndor" << renderer;
    LOG(INFO) << "GL Version" << version;
    LOG(INFO) << "GLSL Version" << glsl_version;
}

void CreateGaussianKernel(std::vector<float> &kernel, int kernel_size, float sigma) {
    float sum(0);
    kernel.resize(kernel_size * kernel_size);
    float center = kernel_size / 2.f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            kernel[i * kernel_size + j] =
                exp(-0.5f * (pow((i - center) / sigma, 2.0f) + pow((j - center) / sigma, 2.0f)))
                    / (2 * M_PI * sigma * sigma);
            sum += kernel[i * kernel_size + j];
        }
    }

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            kernel[i * kernel_size + j] /= sum;
        }
    }
}

}   // namespace feh
