#pragma once
namespace feh {
#include <string>
static const std::string maxpool_comp = R"(
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D input_texture;
layout(rgba32f) uniform image2D output_texture;

const int max_padding = 10;
uniform int kernel_size = max_padding;

float max_pool(in ivec2 pos) {
    float max_value = texelFetch(input_texture, pos, 0).r;
//    if (max_value > 0.5) {
//        max_value = 1.0;
//    } else {
//        max_value = 0.0;
//    }

    for (int i = -kernel_size; i <= kernel_size; ++i) {
        for (int j = -kernel_size; j <= kernel_size; ++j) {
            float value = texelFetch(input_texture, pos + ivec2(i, j), 0).r;
            if (value > max_value) {
//                max_value = 1.0;
                max_value = value;
            }
        }
    }
    return max_value;
}

void main() {
    // retrieve local data
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = textureSize(input_texture, 0);

    if (uv.x >= size.x || uv.y >= size.y) return;

    float value = max_pool(uv);
    imageStore(output_texture, uv, vec4(value));
}

)";
}