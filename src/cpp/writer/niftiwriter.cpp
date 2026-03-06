#include "niftiwriter.h"

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <limits>

#include <zlib.h>

namespace bfiocpp {

// ---------------------------------------------------------------------------
// NIFTI-1 header (348 bytes, packed) — same layout as in niftireader.cpp
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct Nifti1Header_W {
    int32_t  sizeof_hdr;
    char     data_type[10];
    char     db_name[18];
    int32_t  extents;
    int16_t  session_error;
    char     regular;
    char     dim_info;
    int16_t  dim[8];
    float    intent_p1;
    float    intent_p2;
    float    intent_p3;
    int16_t  intent_code;
    int16_t  datatype;
    int16_t  bitpix;
    int16_t  slice_start;
    float    pixdim[8];
    float    vox_offset;
    float    scl_slope;
    float    scl_inter;
    int16_t  slice_end;
    char     slice_code;
    char     xyzt_units;
    float    cal_max;
    float    cal_min;
    float    slice_duration;
    float    toffset;
    int32_t  glmax;
    int32_t  glmin;
    char     descrip[80];
    char     aux_file[24];
    int16_t  qform_code;
    int16_t  sform_code;
    float    quatern_b;
    float    quatern_c;
    float    quatern_d;
    float    qoffset_x;
    float    qoffset_y;
    float    qoffset_z;
    float    srow_x[4];
    float    srow_y[4];
    float    srow_z[4];
    char     intent_name[16];
    char     magic[4];
};
#pragma pack(pop)
static_assert(sizeof(Nifti1Header_W) == 348, "NIFTI-1 writer header must be 348 bytes");

// ---------------------------------------------------------------------------
// Map dtype string → (NIFTI datatype code, element size in bytes)
// ---------------------------------------------------------------------------
static std::pair<int16_t, int> dtype_string_to_nifti(const std::string& s) {
    if (s == "uint8")                    return {2,    1};
    if (s == "int16")                    return {4,    2};
    if (s == "int32")                    return {8,    4};
    if (s == "float32" || s == "float")  return {16,   4};
    if (s == "float64" || s == "double") return {64,   8};
    if (s == "int8")                     return {256,  1};
    if (s == "uint16")                   return {512,  2};
    if (s == "uint32")                   return {768,  4};
    if (s == "int64")                    return {1024, 8};
    if (s == "uint64")                   return {1280, 8};
    throw std::runtime_error("NiftiWriterCPP: unsupported dtype \"" + s + "\"");
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
NiftiWriterCPP::NiftiWriterCPP(
    const std::string& fname,
    const std::vector<std::int64_t>& image_shape,
    const std::string& dtype_str,
    const std::string& dimension_order)
    : _fname(fname), _is_gz(false), _closed(false),
      _nx(1), _ny(1), _nz(1), _nt(1),
      _vox_offset(352.0)
{
    // --- Detect file format ---
    if (fname.size() >= 7 && fname.substr(fname.size() - 7) == ".nii.gz") {
        _is_gz = true;
    } else if (fname.size() >= 4 && fname.substr(fname.size() - 4) == ".nii") {
        _is_gz = false;
    } else {
        throw std::runtime_error(
            "NiftiWriterCPP: unsupported extension (expected .nii or .nii.gz): " + fname);
    }

    // --- Validate and parse dimension_order (same logic as TsWriterCPP) ---
    if (dimension_order.size() < 2 || dimension_order.size() > 5) {
        throw std::invalid_argument(
            "NiftiWriterCPP: dimension_order must contain 2–5 characters, got: \""
            + dimension_order + "\"");
    }
    for (char ch : dimension_order) {
        if (ch != 'X' && ch != 'Y' && ch != 'Z' && ch != 'C' && ch != 'T') {
            throw std::invalid_argument(
                "NiftiWriterCPP: dimension_order contains invalid character, only "
                "T/C/Z/Y/X allowed: \"" + dimension_order + "\"");
        }
    }

    auto pos = dimension_order.find('X');
    if (pos == std::string::npos)
        throw std::invalid_argument("NiftiWriterCPP: dimension_order must contain 'X'");
    _x_index = static_cast<int>(pos);

    pos = dimension_order.find('Y');
    if (pos == std::string::npos)
        throw std::invalid_argument("NiftiWriterCPP: dimension_order must contain 'Y'");
    _y_index = static_cast<int>(pos);

    pos = dimension_order.find('Z');
    if (pos != std::string::npos) _z_index.emplace(static_cast<int>(pos));

    pos = dimension_order.find('C');
    if (pos != std::string::npos) _c_index.emplace(static_cast<int>(pos));

    pos = dimension_order.find('T');
    if (pos != std::string::npos) _t_index.emplace(static_cast<int>(pos));

    // --- Extract image dimensions ---
    if (static_cast<int>(image_shape.size()) != static_cast<int>(dimension_order.size())) {
        throw std::invalid_argument(
            "NiftiWriterCPP: image_shape length must match dimension_order length");
    }
    _nx = image_shape[_x_index];
    _ny = image_shape[_y_index];
    if (_z_index.has_value()) _nz = image_shape[_z_index.value()];
    if (_t_index.has_value()) _nt = image_shape[_t_index.value()];

    // NIFTI-1 dim fields are int16_t — validate
    auto check_dim = [](int64_t d, const char* name) {
        if (d <= 0 || d > std::numeric_limits<int16_t>::max())
            throw std::invalid_argument(
                std::string("NiftiWriterCPP: dimension ") + name +
                " = " + std::to_string(d) + " is out of NIFTI-1 range [1, 32767]");
    };
    check_dim(_nx, "X");
    check_dim(_ny, "Y");
    check_dim(_nz, "Z");
    check_dim(_nt, "T");

    // --- Map dtype ---
    auto [nifti_code, elem_size] = dtype_string_to_nifti(dtype_str);
    _nifti_datatype = nifti_code;
    _elem_size      = elem_size;

    // --- Initialise storage ---
    int64_t pixel_bytes = _nx * _ny * _nz * _nt * static_cast<int64_t>(_elem_size);

    if (_is_gz) {
        // Buffer the entire image in memory; flush (compressed) on Close()
        _pixel_buf.assign(static_cast<std::size_t>(pixel_bytes), 0);
    } else {
        // Create the .nii file, write header, then zero-fill the pixel region
        // so that partial WriteImageData calls work without gaps.
        std::ofstream f(_fname, std::ios::binary | std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("NiftiWriterCPP: cannot create file: " + _fname);

        WriteNifti1Header(f);  // writes 348-byte header + 4-byte extension block

        // Zero-fill pixel data region
        const int64_t CHUNK = 65536;
        std::vector<char> zeros(static_cast<std::size_t>(std::min(pixel_bytes, CHUNK)), 0);
        int64_t remaining = pixel_bytes;
        while (remaining > 0) {
            int64_t n = std::min(remaining, CHUNK);
            f.write(zeros.data(), static_cast<std::streamsize>(n));
            remaining -= n;
        }
        if (!f)
            throw std::runtime_error("NiftiWriterCPP: error initialising file: " + _fname);
    }
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
NiftiWriterCPP::~NiftiWriterCPP() {
    try { Close(); } catch (...) {}
}

// ---------------------------------------------------------------------------
// WriteNifti1Header — write 348-byte header + 4-byte extension block
// ---------------------------------------------------------------------------
void NiftiWriterCPP::WriteNifti1Header(std::ostream& out) {
    Nifti1Header_W hdr;
    std::memset(&hdr, 0, sizeof(hdr));

    hdr.sizeof_hdr = 348;
    hdr.regular    = 'r';

    // ndim: minimum 2, go up if Z or T > 1
    int16_t ndim = 2;
    if (_nz > 1) ndim = 3;
    if (_nt > 1) ndim = 4;

    hdr.dim[0] = ndim;
    hdr.dim[1] = static_cast<int16_t>(_nx);
    hdr.dim[2] = static_cast<int16_t>(_ny);
    hdr.dim[3] = static_cast<int16_t>(_nz);
    hdr.dim[4] = static_cast<int16_t>(_nt);
    hdr.dim[5] = 1;
    hdr.dim[6] = 1;
    hdr.dim[7] = 1;

    hdr.datatype   = _nifti_datatype;
    hdr.bitpix     = static_cast<int16_t>(_elem_size * 8);
    hdr.vox_offset = static_cast<float>(_vox_offset);
    hdr.scl_slope  = 0.0f;   // no scaling

    // Unit pixel spacing
    hdr.pixdim[0] = 1.0f;
    hdr.pixdim[1] = 1.0f;
    hdr.pixdim[2] = 1.0f;
    hdr.pixdim[3] = 1.0f;
    hdr.pixdim[4] = 1.0f;

    // Magic for single-file NIfTI ("n+1\0")
    hdr.magic[0] = 'n';
    hdr.magic[1] = '+';
    hdr.magic[2] = '1';
    hdr.magic[3] = '\0';

    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    // 4-byte extension block (all zeros = no extensions)
    const char ext[4] = {0, 0, 0, 0};
    out.write(ext, 4);
}

// ---------------------------------------------------------------------------
// WriteImageData
// ---------------------------------------------------------------------------
void NiftiWriterCPP::WriteImageData(
    const py::array& py_image,
    const Seq& rows,
    const Seq& cols,
    const std::optional<Seq>& layers,
    const std::optional<Seq>& channels,
    const std::optional<Seq>& tsteps)
{
    if (_closed)
        throw std::runtime_error("NiftiWriterCPP: cannot write after Close()");

    // Determine ranges
    long y0 = rows.Start(), y1 = rows.Stop();
    long x0 = cols.Start(), x1 = cols.Stop();
    long z0 = 0, z1 = 0;
    long t0 = 0, t1 = 0;

    if (_z_index.has_value() && layers.has_value()) {
        z0 = layers->Start();
        z1 = layers->Stop();
    }
    if (_t_index.has_value() && tsteps.has_value()) {
        t0 = tsteps->Start();
        t1 = tsteps->Stop();
    }

    // Clamp to valid image bounds
    x0 = std::max(x0, 0L); x1 = std::min(x1, _nx - 1);
    y0 = std::max(y0, 0L); y1 = std::min(y1, _ny - 1);
    z0 = std::max(z0, 0L); z1 = std::min(z1, _nz - 1);
    t0 = std::max(t0, 0L); t1 = std::min(t1, _nt - 1);

    if (x0 > x1 || y0 > y1)
        throw std::runtime_error("NiftiWriterCPP: empty write region");

    // Get raw byte pointer from the numpy array
    py::buffer_info info = py_image.request();
    const uint8_t* src = static_cast<const uint8_t*>(info.ptr);

    long row_elems = x1 - x0 + 1;
    long row_bytes = row_elems * _elem_size;

    if (_is_gz) {
        // Write rows into the pixel buffer (memory-only)
        long input_i = 0;
        for (long it = t0; it <= t1; ++it) {
            for (long iz = z0; iz <= z1; ++iz) {
                for (long iy = y0; iy <= y1; ++iy) {
                    int64_t flat0 = x0 + _nx * (iy + _ny * (iz + _nz * it));
                    int64_t buf_off = flat0 * _elem_size;
                    std::memcpy(_pixel_buf.data() + buf_off,
                                src + input_i * _elem_size,
                                static_cast<std::size_t>(row_bytes));
                    input_i += row_elems;
                }
            }
        }
    } else {
        // Write rows directly into the .nii file at computed byte offsets
        std::fstream f(_fname, std::ios::binary | std::ios::in | std::ios::out);
        if (!f.is_open())
            throw std::runtime_error("NiftiWriterCPP: cannot open for writing: " + _fname);

        long input_i = 0;
        for (long it = t0; it <= t1; ++it) {
            for (long iz = z0; iz <= z1; ++iz) {
                for (long iy = y0; iy <= y1; ++iy) {
                    int64_t flat0 = x0 + _nx * (iy + _ny * (iz + _nz * it));
                    int64_t file_off = static_cast<int64_t>(_vox_offset) + flat0 * _elem_size;
                    f.seekp(static_cast<std::streamoff>(file_off), std::ios::beg);
                    f.write(reinterpret_cast<const char*>(src + input_i * _elem_size),
                            static_cast<std::streamsize>(row_bytes));
                    input_i += row_elems;
                }
            }
        }
        if (!f)
            throw std::runtime_error("NiftiWriterCPP: write error in file: " + _fname);
    }
}

// ---------------------------------------------------------------------------
// Close — flush .nii.gz to disk (no-op for .nii; idempotent)
// ---------------------------------------------------------------------------
void NiftiWriterCPP::Close() {
    if (_closed) return;
    _closed = true;
    if (_is_gz) {
        FlushGzip();
    }
}

// ---------------------------------------------------------------------------
// FlushGzip — compress header + _pixel_buf and write to the .nii.gz file
// ---------------------------------------------------------------------------
void NiftiWriterCPP::FlushGzip() {
    // Build the 352-byte uncompressed header + extension block
    std::ostringstream hdr_oss;
    WriteNifti1Header(hdr_oss);
    std::string hdr_bytes = hdr_oss.str();
    assert(hdr_bytes.size() == 352);

    // Open output file
    std::ofstream fout(_fname, std::ios::binary | std::ios::trunc);
    if (!fout)
        throw std::runtime_error("NiftiWriterCPP: cannot create .nii.gz file: " + _fname);

    // Initialise gzip deflate stream
    z_stream zs{};
    if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                     16 + 15, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error("NiftiWriterCPP: deflateInit2 failed");
    }

    const uInt OUT_SZ = 65536;
    std::vector<Bytef> out_buf(OUT_SZ);

    // Helper: feed a block of bytes through deflate, writing output to fout
    auto deflate_block = [&](const uint8_t* data, std::size_t len, int flush_mode) {
        const uint8_t* ptr = data;
        std::size_t rem = len;

        // Feed in uInt-sized chunks (handles >4 GB gracefully)
        do {
            uInt chunk = static_cast<uInt>(
                std::min(rem, static_cast<std::size_t>(std::numeric_limits<uInt>::max())));
            zs.next_in  = const_cast<Bytef*>(ptr);
            zs.avail_in = chunk;
            ptr += chunk;
            rem -= chunk;

            int cur_flush = (rem == 0) ? flush_mode : Z_NO_FLUSH;
            int ret;
            do {
                zs.next_out  = out_buf.data();
                zs.avail_out = OUT_SZ;
                ret = deflate(&zs, cur_flush);
                if (ret == Z_STREAM_ERROR) {
                    deflateEnd(&zs);
                    throw std::runtime_error("NiftiWriterCPP: deflate stream error");
                }
                uInt have = OUT_SZ - zs.avail_out;
                fout.write(reinterpret_cast<char*>(out_buf.data()),
                           static_cast<std::streamsize>(have));
            } while (zs.avail_out == 0);
        } while (rem > 0);
    };

    // Compress header, then pixel data
    deflate_block(reinterpret_cast<const uint8_t*>(hdr_bytes.data()),
                  hdr_bytes.size(), Z_NO_FLUSH);
    deflate_block(_pixel_buf.data(), _pixel_buf.size(), Z_FINISH);

    deflateEnd(&zs);

    if (!fout)
        throw std::runtime_error("NiftiWriterCPP: write error creating .nii.gz: " + _fname);
}

} // namespace bfiocpp
