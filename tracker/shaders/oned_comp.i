#pragma once
namespace feh {
#include <string>
static const std::string oned_comp = R"(
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
// SSBO (Shader Storage Buffer Object)
layout(std430, binding=0) buffer EdgeListLayout {
    float edgelist[];
};
layout(std430, binding=1) buffer EvidenceLayout {
    uint evidence[];
};
layout(std430, binding=2) buffer EvidenceDirLayout {
    float evidence_dir[];
};
layout(std430, binding=3) buffer ScoreAndCornerLayout {
    float ratio, distance;
    float tl_x, tl_y, br_x, br_y;
};

// Atomic Object
// reference on atomic counter:
// https://www.khronos.org/opengl/wiki/Atomic_Counter
layout(binding=0) uniform atomic_uint edgepixel_counter;
layout(binding=1) uniform atomic_uint edgepixel_match_counter;

// the sampler, operating on the rendererd depth image
uniform sampler2D this_texture;

// near and far plane, tuning parameters
uniform float z_near = 0.05;
uniform float z_far = 5.0;

// CONSTANTS
const float eps = 1e-4;
const float threshold = 0.1;

// tuning parameters
uniform int search_line_length = 40;   // magic number here,
uniform int intensity_thresh = 128;
uniform float direction_thresh = 0.8;

// convert normalized depth to actual depth
// reference:
// https://www.opengl.org/discussion_boards/showthread.php/145308-Depth-Buffer-How-do-I-get-the-pixel-s-Z-coord
// http://web.archive.org/web/20130416194336/http://olivers.posterous.com/linear-depth-in-glsl-for-real
float linearize_depth(in float z) {
    if (z == 1.0) return -1;
    return 2.0 * z_near * z_far / (z_far + z_near - (2.0 * z - 1) * (z_far - z_near));
}

ivec2 bresenham(ivec2 uv1, ivec2 uv2, float dir, ivec2 size) {
    bool steep = false;
    if (abs(uv1.x-uv2.x) < abs(uv1.y-uv2.y)) {
        uv1 = ivec2(uv1.y, uv1.x);
        uv2 = ivec2(uv2.y, uv2.x);
        steep = true;
    }
    if (uv1.x > uv2.x) {
        ivec2 tmp = uv1;
        uv1 = uv2;
        uv2 = tmp;
    }
    int dx = uv2.x - uv1.x;
    int count = dx + 1;
    int s = 1;
    if (uv2.y < uv1.y) {
        s = -1;
    }
    int dy = s * (uv2.y - uv1.y);
    int dx2 = dx + dx;
    int dy2 = dy + dy;
    int err = 0;
    int x = uv1.x;
    int y = uv1.y;

    while (x <= uv2.x) {
        ivec2 pos = ivec2(x, y);
        if (steep) {
            pos = ivec2(y, x);
        }
        if (evidence[pos.y * size.x + pos.x] >= intensity_thresh) {
            float target_dir = evidence_dir[pos.y * size.x + pos.x];
            if (abs(cos(dir - target_dir)) >= direction_thresh) {
                return pos;
            }
        }
        // proceed
        if (err > dx) {
            y += s;
            err -= dx2;
        }
        x += 1;
        err += dy2;
    }
    return ivec2(-1, -1);
}

float oned_search(ivec2 uv0, float dir, ivec2 size) {
    int cols = size.x;
    int rows = size.y;
    bool found_match = false;

    float cos_th = cos(dir);
    float sin_th = sin(dir);

    if (cos_th < 0) {
        cos_th = -cos_th;
        sin_th = -sin_th;
    }

    float l1 = min(search_line_length, (cols-1-uv0.x)/(cos_th+eps));
    if (sin_th > 0) {
        l1 = min(l1, (rows-1-uv0.y)/(sin_th+eps));
    } else {
        l1 = min(l1, (uv0.y-1)/(-sin_th+eps));
    }
    ivec2 uv1 = ivec2(uv0.x+l1*cos_th, uv0.y+l1*sin_th);
    ivec2 best_match = bresenham(uv0, uv1, dir, size);
    float l2 = search_line_length;
    if (!(best_match.x == -1 && best_match.y == -1)) {
        found_match = true;
        l2 = min(l2, length(best_match - uv0));
    }
    l2 = min(l2, (uv0.x-1) / (cos_th + eps));
    if (sin_th > 0) {
        l2 = min(l2, (uv0.y-1) / (sin_th + eps));
    } else {
        l2 = min(l2, (rows-1-uv0.y) / (-sin_th + eps));
    }
    ivec2 uv2 = ivec2(uv0.x - l2 * cos_th, uv0.y - l2 * sin_th);
    ivec2 best_match2 = bresenham(uv0, uv2, dir, size);
    if (best_match2.x == -1 && best_match2.y == -1) {
        // NO match found in second pass
        if (found_match) {
            return length(best_match - uv0);
        }
    } else {
        return length(best_match2 - uv0);
    }
    return -1;
}

void compute_edge_info(ivec2 pos, ivec2 size) {
    float value[9];
    value[0] = linearize_depth(texelFetch(this_texture, pos+ivec2(-1,-1), 0).r);
    value[1] = linearize_depth(texelFetch(this_texture, pos+ivec2(-1, 0), 0).r);
    value[2] = linearize_depth(texelFetch(this_texture, pos+ivec2(-1,+1), 0).r);
    value[3] = linearize_depth(texelFetch(this_texture, pos+ivec2(0, -1), 0).r);
    value[4] = linearize_depth(texelFetch(this_texture, pos+ivec2(0,  0), 0).r);
    value[5] = linearize_depth(texelFetch(this_texture, pos+ivec2(0, +1), 0).r);
    value[6] = linearize_depth(texelFetch(this_texture, pos+ivec2(+1,-1), 0).r);
    value[7] = linearize_depth(texelFetch(this_texture, pos+ivec2(+1, 0), 0).r);
    value[8] = linearize_depth(texelFetch(this_texture, pos+ivec2(+1,+1), 0).r);
    float delta = 0.25*(abs(value[1]-value[7]) + abs(value[5]-value[3]) + abs(value[0]-value[8]) + abs(value[2]-value[6]));

    if (value[4] != -1 && delta >= threshold) {
        // fill in edgelist
        uint current_index = atomicCounterIncrement(edgepixel_counter);

        edgelist[current_index*4 + 0] = float(pos.x);
        edgelist[current_index*4 + 1] = float(pos.y);
        float dy = -(3*value[0]  - 3*value[2] + 10*value[3] - 10*value[5] + 3*value[6] - 3*value[8]);
        float dx = -(3*value[0]  + 10*value[1] + 3*value[2] - 3*value[6] - 10*value[7] - 3*value[8]);
        edgelist[current_index*4 + 2] = atan(dy, dx);
        float match_dist = oned_search(pos, edgelist[current_index*4 + 2], size);
        edgelist[current_index*4 + 3] = match_dist;

        if (match_dist >= 0) {
            atomicCounterIncrement(edgepixel_match_counter);
        }

    }
}

void main() {
//    uvec3 groups = gl_NumWorkGroups;
//    if (groups.x * groups.y * groups.z == 1) {
//        ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
//        if (uv.x == 1 && uv.y == 1) {
//            // reduction
//            tl_x = 10000;
//            tl_y = 10000;
//            br_x = 0;
//            br_y = 0;
//            uint size = atomicCounter(edgepixel_counter);
//            uint match_size = atomicCounter(edgepixel_match_counter);
//            float total_dist = 0;
//            for (uint i = 0; i < size; ++i) {
//                // accumulate matching distance
//                if (edgelist[size*4+3] >= 0) {
//                    total_dist += edgelist[size*4+3];
//                }
//                // update bounding box corners
//                float x = edgelist[size*4];
//                float y = edgelist[size*4+1];
//                tl_x = min(tl_x, x);
//                tl_y = min(tl_y, y);
//                br_x = max(br_x, x);
//                br_y = max(br_y, y);
//            }
//            distance = total_dist / (match_size + eps);
//            ratio = match_size / (size + eps);
//            ivec2 img_size = textureSize(this_texture, 0);
//            tl_x = max(0, tl_x);
//            tl_y = max(0, tl_y);
//            br_x = min(img_size.x-1, br_x);
//            br_y = min(img_size.y-1, br_y);
//        }
//    } else

    {
        ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
        ivec2 size = textureSize(this_texture, 0);
        if (uv.x < size.x-1 && uv.y < size.y-1
        && uv.x >= 1 && uv.y >= 1) {
            compute_edge_info(uv, size);
        }
    }
}

)";
}