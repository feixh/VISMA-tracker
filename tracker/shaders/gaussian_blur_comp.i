#pragma once
namespace feh {
#include <string>
static const std::string gaussian_blur_comp = R"(
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D input_texture;
layout(rgba32f) uniform image2D output_texture;

const int max_kernel_size = 21;
uniform int kernel_size = max_kernel_size;
uniform float kernel[max_kernel_size * max_kernel_size];

float gaussian_kernel(in ivec2 pos) {
    float sum = 0;
    int half_kernel_size = (kernel_size >> 1);
    for (int i = 0; i <= kernel_size; ++i) {
        for (int j = 0; j <= kernel_size; ++j) {
            float value = texelFetch(input_texture, pos + ivec2(i-half_kernel_size, j-half_kernel_size), 0).r;
            sum += value * kernel[i * kernel_size + j];
        }
    }
    return sum;
}

void main() {
    // retrieve local data
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = textureSize(input_texture, 0);

    if (uv.x >= size.x || uv.y >= size.y) return;

    float value = gaussian_kernel(uv);
    imageStore(output_texture, uv, vec4(value));
}

)";
}