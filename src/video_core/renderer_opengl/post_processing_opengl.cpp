// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/string_util.h"
#include "video_core/renderer_opengl/post_processing_opengl.h"

#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

namespace OpenGL {

// The Dolphin shader header is added here for drop-in compatibility with most
// of Dolphin's "glsl" shaders, which use hlsl types, hence the #define's below
// It's fairly complete, but the features it's missing are:
// The font texture for the ascii shader (Citra doesn't have an overlay font)
// GetTime (not used in any shader provided by Dolphin)
// GetOption* (used in only one shader provided by Dolphin; would require more
// configuration/frontend work)
constexpr char dolphin_shader_header[] = R"(

// hlsl to glsl types
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define uint2 uvec2
#define uint3 uvec3
#define uint4 uvec4
#define int2 ivec2
#define int3 ivec3
#define int4 ivec4

// hlsl to glsl function translation
#define frac fract
#define lerp mix

// Output variable
layout (location = 0) out float4 color;
// Input coordinates
layout (location = 0) in float2 frag_tex_coord;
// Resolution
uniform float4 i_resolution;
uniform float4 o_resolution;
// Layer
uniform int layer;

uniform sampler2D color_texture;
uniform sampler2D color_texture_r;

// Interfacing functions
float4 Sample()
{
    return texture(color_texture, frag_tex_coord);
}

float4 SampleLocation(float2 location)
{
    return texture(color_texture, location);
}

float4 SampleLayer(int layer)
{
    if(layer == 0)
        return texture(color_texture, frag_tex_coord);
    else
        return texture(color_texture_r, frag_tex_coord);
}

#define SampleOffset(offset) textureOffset(color_texture, frag_tex_coord, offset)

float2 GetResolution()
{
    return i_resolution.xy;
}

float2 GetInvResolution()
{
    return i_resolution.zw;
}

float2 GetIResolution()
{
    return i_resolution.xy;
}

float2 GetIInvResolution()
{
    return i_resolution.zw;
}

float2 GetWindowResolution()
{
  return o_resolution.xy;
}

float2 GetInvWindowResolution()
{
  return o_resolution.zw;
}

float2 GetOResolution()
{
    return o_resolution.xy;
}

float2 GetOInvResolution()
{
    return o_resolution.zw;
}

float2 GetCoordinates()
{
    return frag_tex_coord;
}

void SetOutput(float4 color_in)
{
    color = color_in;
}

)";

constexpr char xbr_lv3_builtin[] = R"GLSL(

// xBR-lv3 (Hyllian) adaptado para o Azahar. Licenca MIT original: ver cabecalho do autor.
#define XBR_Y_WEIGHT 48.0
#define XBR_EQ_THRESHOLD 10.0
#define XBR_EQ_THRESHOLD2 2.0
#define XBR_LV2_COEFFICIENT 2.0
#define corner_type 3.0

const vec3 lw = vec3(0.299, 0.587, 0.114) * XBR_Y_WEIGHT;
const vec4 delta = vec4(0.4, 0.4, 0.4, 0.4);

vec4 df(vec4 A, vec4 B)
{
	return vec4(abs(A-B));
}

float c_df(vec3 c1, vec3 c2) {
	vec3 df = abs(c1 - c2);
	return df.r + df.g + df.b;
}

bvec4 eq(vec4 A, vec4 B)
{
	return lessThan(df(A, B), vec4(XBR_EQ_THRESHOLD));
}

bvec4 eq2(vec4 A, vec4 B)
{
	return lessThan(df(A, B), vec4(XBR_EQ_THRESHOLD2));
}

bvec4 band(bvec4 A, bvec4 B)
{
	return bvec4(A.x && B.x, A.y && B.y, A.z && B.z, A.w && B.w);
}

bvec4 bor(bvec4 A, bvec4 B)
{
	return bvec4(A.x || B.x, A.y || B.y, A.z || B.z, A.w || B.w);
}

vec4 weighted_distance(vec4 a, vec4 b, vec4 c, vec4 d, vec4 e, vec4 f, vec4 g, vec4 h)
{
	return (df(a,b) + df(a,c) + df(d,e) + df(d,f) + 4.0*df(g,h));
}

void main()
{
	bvec4 edr, edr_left, edr_up, edr3_left, edr3_up, px; // px = pixel, edr = edge detection rule
	bvec4 interp_restriction_lv1, interp_restriction_lv2_left, interp_restriction_lv2_up;
	bvec4 interp_restriction_lv3_left, interp_restriction_lv3_up;
	bvec4 nc, nc30, nc60, nc45, nc15, nc75; // new_color
	vec4 fx, fx_left, fx_up, finalfx, fx3_left, fx3_up; // inequations of straight lines.
	vec3 res1, res2;
	vec3 pix1 = vec3(0.0), pix2 = vec3(0.0);
	float blend1 = 0.0, blend2 = 0.0;


    vec2 sz = i_resolution.xy;
    vec2 ps = i_resolution.zw;
    float dx = ps.x;
    float dy = ps.y;
    vec2 fp = fract(frag_tex_coord * sz);
    vec2 base = (floor(frag_tex_coord * sz) + vec2(0.5)) * ps;

    vec4 t1 = base.xxxy + vec4(-dx, 0.0, dx, -2.0*dy);
    vec4 t2 = base.xxxy + vec4(-dx, 0.0, dx,     -dy);
    vec4 t3 = base.xxxy + vec4(-dx, 0.0, dx,     0.0);
    vec4 t4 = base.xxxy + vec4(-dx, 0.0, dx,      dy);
    vec4 t5 = base.xxxy + vec4(-dx, 0.0, dx,  2.0*dy);
    vec4 t6 = base.xyyy + vec4(-2.0*dx, -dy, 0.0, dy);
    vec4 t7 = base.xyyy + vec4( 2.0*dx, -dy, 0.0, dy);

    vec3 E = texture(color_texture, t3.yw).rgb;
    vec3 B = texture(color_texture, t2.yw).rgb;
    vec3 D = texture(color_texture, t3.xw).rgb;
    vec3 F = texture(color_texture, t3.zw).rgb;
    vec3 H = texture(color_texture, t4.yw).rgb;
	vec4 e = vec4(dot(E, lw));
	vec4 b = vec4(dot(B, lw), dot(D, lw), dot(H, lw), dot(F, lw));
	if (all(equal(b, e))) { color = vec4(E, 1.0); return; }
    vec3 A1 = texture(color_texture, t1.xw).rgb;
    vec3 B1 = texture(color_texture, t1.yw).rgb;
    vec3 C1 = texture(color_texture, t1.zw).rgb;
    vec3 A = texture(color_texture, t2.xw).rgb;
    vec3 C = texture(color_texture, t2.zw).rgb;
    vec3 G = texture(color_texture, t4.xw).rgb;
    vec3 I = texture(color_texture, t4.zw).rgb;
    vec3 G5 = texture(color_texture, t5.xw).rgb;
    vec3 H5 = texture(color_texture, t5.yw).rgb;
    vec3 I5 = texture(color_texture, t5.zw).rgb;
    vec3 A0 = texture(color_texture, t6.xy).rgb;
    vec3 D0 = texture(color_texture, t6.xz).rgb;
    vec3 G0 = texture(color_texture, t6.xw).rgb;
    vec3 C4 = texture(color_texture, t7.xy).rgb;
    vec3 F4 = texture(color_texture, t7.xz).rgb;
    vec3 I4 = texture(color_texture, t7.xw).rgb;
	vec4 c = vec4(dot(C, lw), dot(A, lw), dot(G, lw), dot(I, lw));
	vec4 d = b.yzwx;
	vec4 f = b.wxyz;
	vec4 g = c.zwxy;
	vec4 h = b.zwxy;
	vec4 i = c.wxyz;

	vec4 i4 = vec4(dot(I4, lw), dot(C1, lw), dot(A0, lw), dot(G5, lw));
	vec4 i5 = vec4(dot(I5, lw), dot(C4, lw), dot(A1, lw), dot(G0, lw));
	vec4 h5 = vec4(dot(H5, lw), dot(F4, lw), dot(B1, lw), dot(D0, lw));
	vec4 f4 = h5.yzwx;

	vec4 c1 = i4.yzwx;
	vec4 g0 = i5.wxyz;
	vec4 b1 = h5.zwxy;
	vec4 d0 = h5.wxyz;

	vec4 Ao = vec4( 1.0, -1.0, -1.0, 1.0 );
	vec4 Bo = vec4( 1.0,  1.0, -1.0,-1.0 );
	vec4 Co = vec4( 1.5,  0.5, -0.5, 0.5 );
	vec4 Ax = vec4( 1.0, -1.0, -1.0, 1.0 );
	vec4 Bx = vec4( 0.5,  2.0, -0.5,-2.0 );
	vec4 Cx = vec4( 1.0,  1.0, -0.5, 0.0 );
	vec4 Ay = vec4( 1.0, -1.0, -1.0, 1.0 );
	vec4 By = vec4( 2.0,  0.5, -2.0,-0.5 );
	vec4 Cy = vec4( 2.0,  0.0, -1.0, 0.5 );

	vec4 Az = vec4( 6.0, -2.0, -6.0, 2.0 );
	vec4 Bz = vec4( 2.0, 6.0, -2.0, -6.0 );
	vec4 Cz = vec4( 5.0, 3.0, -3.0, -1.0 );
	vec4 Aw = vec4( 2.0, -6.0, -2.0, 6.0 );
	vec4 Bw = vec4( 6.0, 2.0, -6.0,-2.0 );
	vec4 Cw = vec4( 5.0, -1.0, -3.0, 3.0 );

	fx      = (Ao*fp.y+Bo*fp.x); 
	fx_left = (Ax*fp.y+Bx*fp.x);
	fx_up   = (Ay*fp.y+By*fp.x);
	fx3_left= (Az*fp.y+Bz*fp.x);
	fx3_up  = (Aw*fp.y+Bw*fp.x);

	// It uses CORNER_C if none of the others are defined.
	{interp_restriction_lv1 = band(band(notEqual(e, f), notEqual(e, h)),
	                             bor(bor(band(not(eq(f,b)), not(eq(f,c))),
	                                   band(not(eq(h,d)), not(eq(h,g)))),
	                                bor(band(eq(e,i), bor(band(not(eq(f,f4)), not(eq(f,i4))),
	                                                   band(not(eq(h,h5)), not(eq(h,i5))))),
	                                   bor(eq(e,g), eq(e,c)))));}

    interp_restriction_lv2_left = band(notEqual(e, g), notEqual(d, g));
	interp_restriction_lv2_up   = band(notEqual(e, c), notEqual(b, c));
	interp_restriction_lv3_left = band(eq2(g,g0), not(eq2(d0,g0)));
	interp_restriction_lv3_up   = band(eq2(c,c1), not(eq2(b1,c1)));

	vec4 fx45 = smoothstep(Co - delta, Co + delta, fx);
	vec4 fx30 = smoothstep(Cx - delta, Cx + delta, fx_left);
	vec4 fx60 = smoothstep(Cy - delta, Cy + delta, fx_up);
	vec4 fx15 = smoothstep(Cz - delta, Cz + delta, fx3_left);
	vec4 fx75 = smoothstep(Cw - delta, Cw + delta, fx3_up);

	edr = band(lessThan(weighted_distance( e, c, g, i, h5, f4, h, f), weighted_distance( h, d, i5, f, i4, b, e, i)), interp_restriction_lv1);
	edr_left = band(lessThanEqual((XBR_LV2_COEFFICIENT*df(f,g)), df(h,c)), interp_restriction_lv2_left);
	edr_up   = band(greaterThanEqual(df(f,g), (XBR_LV2_COEFFICIENT*df(h,c))), interp_restriction_lv2_up);
	edr3_left = interp_restriction_lv3_left;
	edr3_up = interp_restriction_lv3_up;

	nc45 = band(edr, bvec4(fx45));
	nc30 = band(edr, band(edr_left, bvec4(fx30)));
	nc60 = band(edr, band(edr_up, bvec4(fx60)));
	nc15 = band(band(edr, edr_left), band(edr3_left, bvec4(fx15)));
	nc75 = band(band(edr, edr_up), band(edr3_up, bvec4(fx75)));

    px = lessThanEqual(df(e, f), df(e, h));

    nc = bvec4(nc75.x || nc15.x || nc30.x || nc60.x || nc45.x, nc75.y || nc15.y || nc30.y || nc60.y || nc45.y, nc75.z || nc15.z || nc30.z || nc60.z || nc45.z, nc75.w || nc15.w || nc30.w || nc60.w || nc45.w);

	vec4 final45 = vec4(nc45) * fx45;
	vec4 final30 = vec4(nc30) * fx30;
	vec4 final60 = vec4(nc60) * fx60;
	vec4 final15 = vec4(nc15) * fx15;
	vec4 final75 = vec4(nc75) * fx75;

	vec4 maximo = max(max(max(final15, final75),max(final30, final60)), final45);

	     if (nc.x) {pix1 = px.x ? F : H; blend1 = maximo.x;}
	else if (nc.y) {pix1 = px.y ? B : F; blend1 = maximo.y;}
	else if (nc.z) {pix1 = px.z ? D : B; blend1 = maximo.z;}
	else if (nc.w) {pix1 = px.w ? H : D; blend1 = maximo.w;}

	     if (nc.w) {pix2 = px.w ? H : D; blend2 = maximo.w;}
	else if (nc.z) {pix2 = px.z ? D : B; blend2 = maximo.z;}
	else if (nc.y) {pix2 = px.y ? B : F; blend2 = maximo.y;}
	else if (nc.x) {pix2 = px.x ? F : H; blend2 = maximo.x;}

    res1 = mix(E, pix1, blend1);
    res2 = mix(E, pix2, blend2);
    vec3 res = mix(res1, res2, step(c_df(E, res1), c_df(E, res2)));

    color = vec4(res, 1.0);
} 
}
)GLSL";

std::vector<std::string> GetPostProcessingShaderList(bool anaglyph) {
    std::string shader_dir = FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir);
    std::vector<std::string> shader_names;

    if (!FileUtil::IsDirectory(shader_dir)) {
        FileUtil::CreateDir(shader_dir);
    }

    if (anaglyph) {
        shader_dir = shader_dir + "anaglyph";
        if (!FileUtil::IsDirectory(shader_dir)) {
            FileUtil::CreateDir(shader_dir);
        }
    }

    // Would it make more sense to just add a directory list function to FileUtil?
    const auto callback = [&shader_names](u64* num_entries_out, const std::string& directory,
                                          const std::string& virtual_name) -> bool {
        const std::string physical_name = directory + DIR_SEP + virtual_name;
        if (!FileUtil::IsDirectory(physical_name)) {
            // The following is done to avoid coupling this to Qt
            std::size_t dot_pos = virtual_name.rfind(".");
            if (dot_pos != std::string::npos) {
                if (Common::ToLower(virtual_name.substr(dot_pos + 1)) == "glsl") {
                    shader_names.push_back(virtual_name.substr(0, dot_pos));
                }
            }
        }
        return true;
    };

    FileUtil::ForeachDirectoryEntry(nullptr, shader_dir, callback);

    std::sort(shader_names.begin(), shader_names.end());

    return shader_names;
}

std::string GetPostProcessingShaderCode(bool anaglyph, std::string_view shader) {
    if (!anaglyph && shader == "xbr-lv3 (builtin)") {
        return std::string(dolphin_shader_header) + xbr_lv3_builtin;
    }
    std::string shader_dir = FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir);
    std::string shader_path;

    if (anaglyph) {
        shader_dir = shader_dir + "anaglyph";
    }

    // Examining the directory is done because the shader extension might have an odd case
    // This can be eliminated if it is specified that the shader extension must be lowercase
    const auto callback = [&shader, &shader_path](u64* num_entries_out,
                                                  const std::string& directory,
                                                  const std::string& virtual_name) -> bool {
        const std::string physical_name = directory + DIR_SEP + virtual_name;
        if (!FileUtil::IsDirectory(physical_name)) {
            // The following is done to avoid coupling this to Qt
            std::size_t dot_pos = virtual_name.rfind(".");
            if (dot_pos != std::string::npos) {
                if (Common::ToLower(virtual_name.substr(dot_pos + 1)) == "glsl" &&
                    virtual_name.substr(0, dot_pos) == shader) {
                    shader_path = physical_name;
                    return false;
                }
            }
        }
        return true;
    };

    FileUtil::ForeachDirectoryEntry(nullptr, shader_dir, callback);
    if (shader_path.empty()) {
        return "";
    }

    boost::iostreams::stream<boost::iostreams::file_descriptor_source> file;
    FileUtil::OpenFStream<std::ios_base::in>(file, shader_path);
    if (!file.is_open()) {
        return "";
    }

    std::stringstream shader_text;
    shader_text << file.rdbuf();

    return dolphin_shader_header + shader_text.str();
}

} // namespace OpenGL
