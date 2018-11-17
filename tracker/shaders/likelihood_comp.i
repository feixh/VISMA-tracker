#pragma once
namespace feh {
#include <string>
static const std::string likelihood_comp = R"(
// compute shader for likelihood evaluation
#version 430 core

layout(local_size_x = 1) in;
layout(std430, binding=0) buffer ColumnSumsLayout {
    float column_sums[];
};
layout(binding=1) uniform atomic_uint counter_edgepixels;


uniform int likelihood_type = 0;    // 0 for intersection kernel and 1 for cross entropy
uniform float edge_ratio = 0.01;
const float EPS = 1e-4;

uniform sampler2D evidence_texture;
uniform sampler2D prediction_texture;

float intersection_kernel(in float prediction, in float evidence) {
    if (prediction > 0.5) {
        atomicCounterIncrement(counter_edgepixels);
    }
    return min(prediction, evidence);
}

float cross_entropy(in float prediction, in float evidence) {
    float value = (1-edge_ratio) * evidence * log(EPS + prediction)
                    + edge_ratio * (1 - evidence) * log(EPS + 1 - prediction);
    return -value;
}

void main() {
    uint col = gl_WorkGroupID.x;
    float sum = 0;
    ivec2 uv;
    ivec2 size = textureSize(prediction_texture, 0);
    for (int i = 0; i < size.y; ++i) {
        uv = ivec2(i, col);
        if (likelihood_type == 0) {
            sum += intersection_kernel(texelFetch(prediction_texture, uv, 0).r, texelFetch(evidence_texture, uv, 0).r);
        } else {
            sum += cross_entropy(texelFetch(prediction_texture, uv, 0).r, texelFetch(evidence_texture, uv, 0).r);
        }
    }
    column_sums[col] = sum;
}


)";
}