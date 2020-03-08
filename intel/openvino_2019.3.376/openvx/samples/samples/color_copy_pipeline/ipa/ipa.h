/*
        Copyright 2019 Intel Corporation.
        This software and the related documents are Intel copyrighted materials,
        and your use of them is governed by the express license under which they
        were provided to you (End User License Agreement for the Intel(R) Software
        Development Products (Version May 2017)). Unless the License provides
        otherwise, you may not use, modify, copy, publish, distribute, disclose or
        transmit this software or the related documents without Intel's prior
        written permission.

        This software and the related documents are provided as is, with no
        express or implied warranties, other than those that are expressly
        stated in the License.
*/
/* Intel Printing Acceleration library */

#ifndef ipa_h_INCLUDED

#define ipa_h_INCLUDED

#include <stddef.h>

#ifdef  __cplusplus
extern "C"
{
#endif

/* IPA data type definitions */

/* restrict is standard in C99, but not in all C++ compilers. */
#ifdef _MACOS
#define ipa_restrict __restrict
#elif __STDC_VERSION__ == 199901L /* C99 */
#elif defined(_MSC_VER) && _MSC_VER >= 1500 /* MSVC 9 or newer */
#define ipa_restrict __restrict
#elif __GNUC__ >= 3 /* GCC 3 or newer */
#define ipa_restrict __restrict
#else /* Unknown or ancient */
#define ipa_restrict
#endif

typedef float ipa_float;
typedef unsigned char ipa_byte;

typedef unsigned char ipa_bool;
#define ipa_false                      0
#define ipa_true                       1


/* Library instantiation */

/* Opaque library instance type */
typedef struct ipa_context_s ipa_context;

/* Callers pass us details of allocator functions */
typedef struct
{
    /* These functions are assumed to return blocks aligned to
     * sizeof(void *) at least. */
    void *(*ipa_malloc)(void *opaque, size_t size);
    void *(*ipa_realloc)(void *opaque, void *ptr, size_t newsize);
    void  (*ipa_free)(void *opaque, void *ptr);
} ipa_allocators;

/* Initialise an instance. */
ipa_context *ipa_init(const ipa_allocators *, void *opaque);

/* Finalise an instance. */
void ipa_fin(ipa_context *, void *opaque);

/* Force the IPA libraries use of SSE on (1) or off (0).
 * Returns non-zero if the forcing did not work (i.e. if the
 * library does not contain the required cores). */
int ipa_force_sse(ipa_context *, int sse);

/* Determine if the CPU supports SSE or not. */
int ipa_cpu_supports_sse_4_1(ipa_context *);

/* Operations */

/* Image Rescaling: ************************************************

Scale a src_w*src_h bitmap to a dst_w*dst_h one, and return the
(patch_x, patch_y) + (patch_w,patch_h) rectangle of the final image.

0 <= patch_x, patch_x + patch_w <= dst_w
0 <= patch_y, patch_y + patch_h <= dst_h

In many cases callers will use patch_x = patch_y = 0, and
dst_w = patch_w, dst_h = patch_h to get the whole output.

For some applications (typically due to banding), the caller may
wish to only supply a subset of the image data. The data_x,data_y,
data_w,data_h entries give the subregion of the source data that
is actually supplied.

In many cases callers will use data_x = data_y = 0, and
data_w = src_w, data_h = dst_w and just supply all the data.

Data is assumed to be 'bpp' bits per pixel, with 'channels' channels.
Only 8bpp, and 1, 3 or 4 channels are supported currently.
*/


/* Which filter function to use:
 *   NEAREST = Nearest neighbour
 *   LINEAR = Linear interpolation
 *   DOGLEG = Dogleg curve for bicubic
 *   MITCHELL = Mitchell curve for bicubic
 */
typedef enum {
    IPA_NEAREST = 0,
    IPA_LINEAR = 1,
    IPA_DOGLEG = 2,
    IPA_MITCHELL = 3
} ipa_rescale_quality;

/* Opaque type for rescalar instance. */
typedef struct ipa_rescaler_s ipa_rescaler;

/* Initialise a rescaler instance. */
ipa_rescaler *ipa_rescaler_init(ipa_context *context, void *opaque,
                                unsigned int src_w, unsigned int src_h,
                                unsigned int data_x, unsigned int data_y,
                                unsigned int data_w, unsigned int data_h,
                                unsigned int dst_w, unsigned int dst_h,
                                unsigned int patch_x, unsigned int patch_y,
                                unsigned int patch_w, unsigned int patch_h,
                                ipa_rescale_quality quality,
                                unsigned int src_bytes_per_channel,
                                unsigned int dst_bytes_per_channel,
                                unsigned int channels,
                                unsigned int src_max_value,
                                unsigned int dst_max_value);

/* Process a scanlines worth of data.
 *
 * Called with:
 *    input  = pointer to a scanline of input data (or NULL)
 *    output = pointer to a place to be filled with a scanline of output data
 * Returns with flag word.
 *    Bit 0 set => input was consumed.
 *    Bit 1 set => output was provided.
 */
int ipa_rescaler_process(ipa_rescaler *, void *opaque, const unsigned char *input, unsigned char *output);

/* Reset a rescaler instance. */
void ipa_rescaler_reset(ipa_rescaler *, void *opaque);

/* Finalise a rescaler instance. */
void ipa_rescaler_fin(ipa_rescaler *, void *opaque);

/* Image Rotation: ************************************************

Given an image of src_w * src_h, prescale it by pre_x_scale,
pre_y_scale, rotate it by t_degrees (-90 <= t_degrees <= 90),
and post-scale it post_x_scale, post_y_scale. The final bitmap
dimensions are placed into dst_w and dst_h.

Any area outside of the the source image will be filled with the
color given in bg.

*/

/* Opaque type for rescalar instance. */
typedef struct ipa_rotator_s ipa_rotator;

void
ipa_rotator_pre_init(unsigned int  src_w,
                     unsigned int  src_h,
                     unsigned int *dst_w,
                     unsigned int *dst_h,
                     double        t_degrees,
                     double        pre_x_scale,
                     double        pre_y_scale,
                     double        post_x_scale,
                     double        post_y_scale);

ipa_rotator *ipa_rotator_init(ipa_context *context, void *opaque,
                              unsigned int src_w, unsigned int src_h,
                              unsigned int *dst_w, unsigned int *dst_h,
                              double t_degrees,
                              double pre_x_scale, double pre_y_scale,
                              double post_x_scale, double post_y_scale,
                              unsigned char *bg,
                              unsigned int channels);

void ipa_rotator_map_band(ipa_rotator  *rotator,
                          unsigned int  dst_y0,
                          unsigned int  dst_y1,
                          unsigned int *src_y0,
                          unsigned int *src_y1,
                          unsigned int *src_w);

int ipa_rotator_band(ipa_rotator         *rotator,
                     void                *opaque,
                     const unsigned char *src,
                     int                  src_stride,
                     unsigned int         src_y0,
                     unsigned int         src_y1,
                     unsigned char       *dst,
                     int                  dst_stride,
                     unsigned int         dst_y0,
                     unsigned int         dst_y1);

void ipa_rotator_reset(ipa_rotator *, void *opaque);

void ipa_rotator_fin(ipa_rotator *, void *opaque);

/* Pixel "doubler" - Image integer rescalers: *********************

 Given an image of src_w * src_h (1 byte per sample, "channels"
 samples per pixel), scale up by an integer factor.

 */

/* Opaque type for doubler instance. */
typedef struct ipa_doubler_s ipa_doubler;

typedef enum {
    IPA_DOUBLE_NEAREST = 0,
    IPA_DOUBLE_INTERP = 1,
    IPA_DOUBLE_MITCHELL = 3
} ipa_double_quality;

/* On exit, in_lines is set to the number of input lines of
 * data that should be presented at once.
 */
ipa_doubler *ipa_doubler_init(ipa_context        *context,
                              void               *opaque,
                              unsigned int        src_w,
                              unsigned int        src_h,
                              unsigned int        factor,
                              ipa_double_quality  quality,
                              unsigned int        channels,
                              unsigned int       *in_lines);

/* This should be called repeatedly, until the required number
 * of destination lines have been received. This will typically
 * take src_h * factor + *in_lines - 1 calls.
 *
 * input should point to an array of 'in_lines' pointers (as
 * returned from ipa_doubler_init), one per input line of data.
 * These should be initialised on entry to point to the last
 * n lines of data, input[0] being oldest, input[n-1] being the
 * newest (NULL if no data).
 *
 * The first call will have input set up to point to a single
 * line of data, and all the other input values being NULL.
 * For the next call, the data pointers are "rolled" backwards,
 * so input[0] = input[1], etc, and the new data pointer
 * is input[(*in_lines)-1]. At the end of the image, data pointers
 * should be set to NULL.
 *
 * output should point to an array of 'factor' pointers to
 * output buffers for the line data.
 *
 * The return code is the number of output lines produced
 * (guaranteed to be <= factor).
 */
int ipa_doubler_process(ipa_doubler *, void *opaque,
                        const unsigned char **input,
                        unsigned char **output);

void ipa_doubler_fin(ipa_doubler *, void *opaque);

/* Structure for data supplied in halftone call back */
typedef struct ipa_halftone_data_s {
    const unsigned char *data;
    int                  offset_x;
    int                  raster;
    int                  x;
    int                  y;
    int                  w;
    int                  h;
    int                  plane_raster;
} ipa_halftone_data_t;

/* pixel (x,y) of plane p value is given by: data[offset_x + (x>>3) + y*raster + p*plane_raster]>>(x & 7) & 1 */

/* Opaque type for halftone instance and dda. */
typedef struct ipa_halftone_s ipa_halftone;
typedef struct ipa_dda_s ipa_dda;
typedef struct ipa_matrix_s {
    float xx; float xy;
    float yx; float yy;
    float tx; float ty;
} ipa_matrix;

typedef void (ipa_ht_callback_t)(ipa_halftone_data_t *data, void *args);

/* Initialize a halftone instance. */
ipa_halftone *ipa_halftone_init(ipa_context       *ctx,
                                void              *opaque,
                                int                w,
                                int                h,
                                const ipa_matrix  *mat,
                                unsigned int       num_planes,
                                unsigned char     *cache,
                                int                clip_x,
                                int                clip_y,
                                int                clip_w,
                                int                clip_h,
                                int                any_part_of_pixel);

/* Reset a halftone so that it can be used again with exactly the
 * same parameters. */
void ipa_halftone_reset(ipa_context *ctx,
                        ipa_halftone *ht);

/* Set a screen for a colorant. */
int ipa_halftone_add_screen(ipa_context   *ctx,
                            void          *opaque,
                            ipa_halftone  *ht,
                            int            invert,
                            unsigned int   width,
                            unsigned int   height,
                            unsigned int   x_phase,
                            unsigned int   y_phase,
                            unsigned char *values);

/* Return non-zero if the next scanlines data will be used. */
/* This allows callers to avoid preparing data for scanlines
 * that will be entirely clipped away, or will not appear
 * due to downscales. If this returns zero, then DO NOT
 * call ipa_halftone_process_planar with the data. */
int ipa_halftone_next_line_required(ipa_halftone *ht);

/* Halftone some data. */
int ipa_halftone_process_planar(ipa_halftone         *ht,
                                void                 *opaque,
                                const unsigned char **buffer,
                                ipa_ht_callback_t    *callback,
                                void                 *callback_arg);

/* Maybe:
int ipa_halftone_process_chunky(ipa_halftone        *ht,
                                void                *opaque,
                                const unsigned char *buffer,
                                ipa_ht_callback_t   *callback,
                                void                *callback_arg);
*/

/* Finalize a halftone instance. */
void ipa_halftone_fin(ipa_halftone *, void *opaque);

/* Transparency blending routines */

/* The following routines are used to perform transparency blending between layers.
 * Transparency operations build a stack of areas to be composited into each other.
 * Each of these routines performs a single step of this process, compositing the
 * top-of-stack (tos) group into the next-on-stack (nos) group.  The operation
 * updates the nos group with the result, while the tos group is left unchanged
 * and can be discarded after composition.
 *
 * Only certain subsets of commonly used blends are supported.  The following
 * routines optimize two tos isolation cases (isolated and non-isolated), combined
 * with three softmask cases: no softmask, a partial mask with a background value
 * for areas outside of the mask, or a full mask which covers the entire area
 * of the blend.
 *
 * The data to be composed is presented in planar format, with n_chan-1 color
 * planes and a final alpha plane.  Planes are stored consecutively into a
 * single data area, with parameters including X/Y bounds, row and plane
 * strides further specifying the group geometry.
 *
 * Parameters:
 *
 *    ctx              - Pointer to an initialized IPA context.
 *    tos_ptr          - Pointer to color planes plus alpha channel data for the
 *                       top-of-stack group.
 *    tos_rowstride    - Number of bytes used for each row of the tos data.
 *                       This value must be >= (x1 - y1) and can include padding.
 *    tos_planestride  - Number of bytes between each tos data plane.
 *    nos_ptr          - Pointer to color planes plus alpha channel data for the
 *                       next-on-stack group.
 *    nos_planestride  - Number of bytes between each nos data plane.
 *    nos_rowstride    - Number of bytes used for each row of the nos data.
 *                       This value must be >= (x1 - y1) and can include padding.
 *    alpha            - scalar value representing an invariant alpha level which
 *                       is applied across the entire raster area.
 *    n_chan           - The total number of channels in both the tos and nos
 *                       groups including all colors but not the alpha channel.
 *                       Storage must be allocated for n_chan + 1 channels.
 *                       The tos and nos groups must have the same number of
 *                       channels.
 *    x0, y0, x1, y1   - A rectangle describing the sub-area of the plane to be
 *                       blended.
 *
 *  The following additional parameters are specified for blends with softmasks
 *  (the partialmask and fullmask cases):
 *
 *    mask_ptr         - Pointer to softmask data raster.
 *    mask_rowstride   - Number of bytes between each row of mask data.
 *    mask_x0, mask_x1,
 *    mask_y0, mask_y1 - For partial masks only, the bounding rectangle
 *                       specifying the area of the mask data to be applied.
 *    mask_bg_alpha    - Scalar invariant mask background alpha value, used
 *                       for any areas which are outside of the softmask
 *                       bounds.
 *    mask_tr_fn       - Pointer to a 256-byte array mapping softmask alpha levels
 *                       to adjusted levels.  Can be null to specify a linear
 *                       transfer function.
 */

extern void compose_blend_nonisolated_nomask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte alpha);

extern void compose_blend_nonisolated_partialmask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte *ipa_restrict mask_ptr,
    int mask_rowstride,
    int mask_x0, int mask_y0, int mask_x1, int mask_y1,
    ipa_byte mask_bg_alpha,
    ipa_byte *ipa_restrict mask_tr_fn,
    ipa_byte alpha);

extern void compose_blend_nonisolated_fullmask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte *ipa_restrict mask_ptr,
    int mask_rowstride,
    ipa_byte mask_bg_alpha,
    ipa_byte *ipa_restrict mask_tr_fn,
    ipa_byte alpha);

extern void compose_blend_isolated_nomask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte alpha);

extern void compose_blend_isolated_partialmask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte *ipa_restrict mask_ptr,
    int mask_rowstride,
    int mask_x0, int mask_y0, int mask_x1, int mask_y1,
    ipa_byte mask_bg_alpha,
    ipa_byte *ipa_restrict mask_tr_fn,
    ipa_byte alpha);

extern void compose_blend_isolated_fullmask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte *ipa_restrict mask_ptr,
    int mask_rowstride,
    ipa_byte mask_bg_alpha,
    ipa_byte *ipa_restrict mask_tr_fn,
    ipa_byte alpha);

/* Gradient blend routines */

/* The render_axial_gradient and render_radial_gradient routines draw
 * areas of axial or radial blends into a rectangular buffer.  These
 * routines  support PS3 and PDF-style blends using four-component
 * 8 bit/component color spaces (RGBx or CMYK).  Start and end extend
 * regions are supported, as well as color function parameter matching
 * through table lookups.
 *
 * Gradients are generated by mathematical equations which map each
 * raster point to a parametric variable t.  If (0.0 <= t <= 1.0),
 * the point lies within the active area of the gradient, and its
 * color is determined by mapping t through a color lookup table.
 * If (t < 0.0) or (t > 1.0), the point falls into the inner or
 * outer extend area respectively, and is either marked with the
 * inner our outer extent color or left unmarked, based on the
 * setting of the corresponding extend flag.
 *
 * Raster coordinates are specified using base, offset and size
 * parameters, as follows:
 *
 * - The data buffer should be large enough to hold at least
 *   (offset_x + size_x) samples in X, and (offset + size_y) samples
 *   in Y.  The rowstride should be >= (offset_x + size_x) * 4 bytes.
 *
 * - The base_x and base_y values specify the mathematical coordinate
 *   of the point (0, 0) in the data buffer.  These values can be used
 *   to render into tiles or bands by setting the base coordinates to
 *   the corner of each tile or band.
 *
 * - The offset and size values specify a rectangular subset of the
 *   data to be rendered.  The offset_x and offset_y values are relative
 *   to the point (0, 0) in the data buffer; e.g. an offset of (1.0, 1.0)
 *   leaves a 1-pixel strip unmarked across the top and left edges of the
 *   image.  The size values specify the dimensions of the subrectangle
 *   to be rendered.
 *
 * The data buffer format is RGBx 8 bits/component (32 bits/sample).
 * This routine can also be used for CMYK color conversions by
 * passing CMYK values in the color table.
 *
 * The color table is a 1-to-4 function which maps the parametric
 * variable t to a 4-element color value.  All input and output
 * values for this table are floating-point numbers in the range
 * 0.0 - 1.0.
 *
 * Parameters:
 *    ctx                  - Pointer to an initialized IPA context.
 *    data_ptr             - Pointer to four-component 8 BPC data buffer.
 *    data_rowstride       - Data buffer rowstride in bytes.
 *    base_x, base_y       - Base coordinates used to position this data buffer. 
 *    offset_x, offset_y   - Offsets (samples) to the subrectangle to be filled.
 *    size_x, int size_y   - Size of the subrectangle to be filled.
 *
 *    axis_start, axis_end - (Axial only) Axis coordinates for the gradient vector.
 *
 *    center               - (Radial only) Center point of inner and outer circles.
 *                           Note that this implementation does not support non-
 *                           concentric circles which produce conical blends.
 *                           Not including the extend areas or rectangle bounds,
 *                           radial blends generated by this routine will always
 *                           have a circular shape (if one radius is 0.0) or a
 *                           donut shape (if both radii are greater than zero).
 *    radius_start         - (Radial only) Radius corresponding to the start color value.
 *    radius_end           - (Radial only) Radius corresponding to the end color value.
 *
 *    color_count          - Number of colors in the lookup table.  This can be
 *                           any number >= 1 and is not limited to a power of two.
 *    color_interpolation  - Flag to enable interpolation between color table values.
 *    colors               - Array of <color_count> sets of four floating-point
 *                           color components in the range 0.0 to 1.0.
 *    extend_start         - Flag to extend the start color to any areas before the
 *                           start axis point.  If unset, the area will be left unmarked.
 *    extend_start         - Flag to extend the end color to any areas after the
 *                           end axis point.  If unset, the area will be left unmarked.
 *    mask_ptr             - If non-NULL, pointer to a buffer containing at least
 *                           size_y * mask_rowstride bytes, which will be set to a mask
 *                           pattern denoting which pixels were marked.  Position [0 0]
 *                           of the mask buffer corresponds to the sample at position
 *                           [offset_x, offset_y] in the gradient buffer, so the mask
 *                           buffer does not need to include any bounds outside of the
 *                           subrectangle which is rendered.  Each byte in the mask buffer
 *                           will be set to 0xFF if the corresponding sample was marked
 *                           in the data buffer, or to 0x00 if the pixel was left unmarked.
 *    mask_rowstride       - Row stride of the mask buffer.  Should be set to zero
 *                           if mask_ptr is NULL.
 *
 * Application Notes:
 *
 * - The mask buffer can be useful when pre-rendering to an intermediate
 *   buffer rather than writing directly to the output raster.  It is also
 *   possible to render directly since any unmarked extend areas will be
 *   left unmodified in the color gradient raster.  Note that if both Extend
 *   flags are set, the entire render subrectangle will always be marked,
 *   so the mask can be omitted in this case for performance.
 *
 * - The color function supports either interpolated or non-interpolated
 *   lookup tables.  PS3/PDF color functions can be discontinous, so if
 *   you wish to avoid color transitions for complex functions, use a
 *   larger color table with interpolation disabled.
 */

#define IPA_GRADIENT_COLOR_COMPONENTS_MAX   4
typedef struct ipa_point {
    ipa_float x;
    ipa_float y;
} ipa_point;

/* Render an axial gradient into a color buffer and optional mask buffer. */
extern void render_axial_gradient(ipa_context *ctx,
                                  ipa_byte *ipa_restrict data_ptr,
                                  int data_rowstride,
                                  int base_x,
                                  int base_y,
                                  int offset_x,
                                  int offset_y,
                                  int size_x,
                                  int size_y,
                                  ipa_point axis_start,
                                  ipa_point axis_end,
                                  int color_count,
                                  ipa_bool color_interpolation,
                                  ipa_float colors[][IPA_GRADIENT_COLOR_COMPONENTS_MAX],
                                  ipa_bool extend_start,
                                  ipa_bool extend_end,
                                  ipa_byte *ipa_restrict mask_ptr,
                                  int mask_rowstride);

/* Render a radial gradient into a color buffer and optional mask buffer. */
extern void render_radial_gradient(ipa_context *ctx,
                                   ipa_byte *ipa_restrict data_ptr,
                                   int data_rowstride,
                                   int base_x,
                                   int base_y,
                                   int offset_x,
                                   int offset_y,
                                   int size_x,
                                   int size_y,
                                   ipa_point center,
                                   ipa_float radius_start,
                                   ipa_float radius_end,
                                   int color_count,
                                   ipa_bool color_interpolation,
                                   ipa_float colors[][IPA_GRADIENT_COLOR_COMPONENTS_MAX],
                                   ipa_bool extend_start,
                                   ipa_bool extend_end,
                                   ipa_byte *ipa_restrict mask_ptr,
                                   int mask_rowstride);

#ifdef  __cplusplus
} // extern "C"
#endif

#endif /* ipa_h_INCLUDED */

