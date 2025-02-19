// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

//? #version 430 core
precision mediump float;

layout(location = 0) in vec2 tex_coord;
layout(location = 0) out vec2 frag_color;

layout(binding = 0) uniform sampler2D tex_input;

const vec3 K = vec3(0.2627, 0.6780, 0.0593);
// TODO: improve handling of alpha channel
#define GetLum(xoffset) dot(K, textureLodOffset(tex_input, tex_coord, 0.0, ivec2(xoffset, 0)).rgb)

void main() {
    float l = GetLum(-1);
    float c = GetLum(0);
    float r = GetLum(1);

    frag_color = vec2(r - l, l + 2.0 * c + r);
}
