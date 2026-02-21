#include "niftireader.h"

#include <stdexcept>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <cmath>

#include <zlib.h>

namespace bfiocpp {

// ---------------------------------------------------------------------------
// NIFTI-1 header (348 bytes, packed)
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct Nifti1Header {
    int32_t  sizeof_hdr;    // 4
    char     data_type[10]; // 14
    char     db_name[18];   // 32
    int32_t  extents;       // 36
    int16_t  session_error; // 38
    char     regular;       // 39
    char     dim_info;      // 40
    int16_t  dim[8];        // 56
    float    intent_p1;     // 60
    float    intent_p2;     // 64
    float    intent_p3;     // 68
    int16_t  intent_code;   // 70
    int16_t  datatype;      // 72
    int16_t  bitpix;        // 74
    int16_t  slice_start;   // 76
    float    pixdim[8];     // 108
    float    vox_offset;    // 112
    float    scl_slope;     // 116
    float    scl_inter;     // 120
    int16_t  slice_end;     // 122
    char     slice_code;    // 123
    char     xyzt_units;    // 124
    float    cal_max;       // 128
    float    cal_min;       // 132
    float    slice_duration;// 136
    float    toffset;       // 140
    int32_t  glmax;         // 144
    int32_t  glmin;         // 148
    char     descrip[80];   // 228
    char     aux_file[24];  // 252
    int16_t  qform_code;    // 254
    int16_t  sform_code;    // 256
    float    quatern_b;     // 260
    float    quatern_c;     // 264
    float    quatern_d;     // 268
    float    qoffset_x;     // 272
    float    qoffset_y;     // 276
    float    qoffset_z;     // 280
    float    srow_x[4];     // 296
    float    srow_y[4];     // 312
    float    srow_z[4];     // 328
    char     intent_name[16];// 344
    char     magic[4];      // 348
};
#pragma pack(pop)
static_assert(sizeof(Nifti1Header) == 348, "NIFTI-1 header must be 348 bytes");

// ---------------------------------------------------------------------------
// NIFTI-2 header (540 bytes, packed)
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct Nifti2Header {
    int32_t  sizeof_hdr;    // 4
    char     magic[8];      // 12
    int16_t  datatype;      // 14
    int16_t  bitpix;        // 16
    int64_t  dim[8];        // 80
    double   intent_p1;     // 88
    double   intent_p2;     // 96
    double   intent_p3;     // 104
    double   pixdim[8];     // 168
    int64_t  vox_offset;    // 176
    double   scl_slope;     // 184
    double   scl_inter;     // 192
    double   cal_max;       // 200
    double   cal_min;       // 208
    double   slice_duration;// 216
    double   toffset;       // 224
    int64_t  slice_start;   // 232
    int64_t  slice_end;     // 240
    char     descrip[80];   // 320
    char     aux_file[24];  // 344
    int32_t  qform_code;    // 348
    int32_t  sform_code;    // 352
    double   quatern_b;     // 360
    double   quatern_c;     // 368
    double   quatern_d;     // 376
    double   qoffset_x;     // 384
    double   qoffset_y;     // 392
    double   qoffset_z;     // 400
    double   srow_x[4];     // 432
    double   srow_y[4];     // 464
    double   srow_z[4];     // 496
    int32_t  slice_code;    // 500
    int32_t  xyzt_units;    // 504
    int32_t  intent_code;   // 508
    char     intent_name[16];// 524
    char     dim_info;      // 525
    char     unused_str[15];// 540
};
#pragma pack(pop)
static_assert(sizeof(Nifti2Header) == 540, "NIFTI-2 header must be 540 bytes");

// ---------------------------------------------------------------------------
// Byte-swap helpers
// ---------------------------------------------------------------------------
static inline void bswap2(void* p) {
    auto* b = reinterpret_cast<uint8_t*>(p);
    std::swap(b[0], b[1]);
}
static inline void bswap4(void* p) {
    auto* b = reinterpret_cast<uint8_t*>(p);
    std::swap(b[0], b[3]);
    std::swap(b[1], b[2]);
}
static inline void bswap8(void* p) {
    auto* b = reinterpret_cast<uint8_t*>(p);
    std::swap(b[0], b[7]);
    std::swap(b[1], b[6]);
    std::swap(b[2], b[5]);
    std::swap(b[3], b[4]);
}

// ---------------------------------------------------------------------------
// Map NIFTI datatype to bfiocpp data_type string
// ---------------------------------------------------------------------------
static std::string nifti_dtype_to_string(int16_t dt) {
    switch (dt) {
        case 2:    return "uint8";
        case 4:    return "int16";
        case 8:    return "int32";
        case 16:   return "float";
        case 64:   return "double";
        case 256:  return "int8";
        case 512:  return "uint16";
        case 768:  return "uint32";
        case 1024: return "int64";
        case 1280: return "uint64";
        default:   return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
NiftiReaderCPP::NiftiReaderCPP(const std::string& fname)
    : _fname(fname)
{
    // Detect gzip by extension
    if (fname.size() >= 7 &&
        fname.substr(fname.size() - 7) == ".nii.gz") {
        _is_gz = true;
    } else if (fname.size() >= 4 &&
               fname.substr(fname.size() - 4) == ".nii") {
        _is_gz = false;
    } else {
        throw std::runtime_error("NiftiReaderCPP: unsupported extension (expected .nii or .nii.gz)");
    }
    ReadHeader();
}

// ---------------------------------------------------------------------------
// ReadHeader — handles NIFTI-1 and NIFTI-2, with byte-swap detection
// ---------------------------------------------------------------------------
void NiftiReaderCPP::ReadHeader() {
    // For .nii.gz we must inflate first to read the header
    if (_is_gz) {
        EnsureDataLoaded();
        if (_raw_data.size() < 4) {
            throw std::runtime_error("NiftiReaderCPP: compressed file too small");
        }

        int32_t sh;
        std::memcpy(&sh, _raw_data.data(), sizeof(sh));

        bool swapped = false;
        int32_t sh_swapped = sh;
        bswap4(&sh_swapped);

        if (sh == 348 || sh == 540) {
            swapped = false;
        } else if (sh_swapped == 348 || sh_swapped == 540) {
            swapped = true;
            sh = sh_swapped;
        } else {
            throw std::runtime_error("NiftiReaderCPP: invalid sizeof_hdr in compressed file");
        }
        _byte_swapped = swapped;

        if (sh == 348) {
            if (_raw_data.size() < 348) {
                throw std::runtime_error("NiftiReaderCPP: compressed file too small for NIFTI-1 header");
            }
            Nifti1Header hdr;
            std::memcpy(&hdr, _raw_data.data(), sizeof(hdr));

            if (_byte_swapped) {
                for (int i = 0; i < 8; ++i) bswap2(&hdr.dim[i]);
                bswap2(&hdr.datatype);
                for (int i = 0; i < 8; ++i) bswap4(&hdr.pixdim[i]);
                bswap4(&hdr.vox_offset);
                bswap4(&hdr.scl_slope);
                bswap4(&hdr.scl_inter);
            }

            int ndim = hdr.dim[0];
            _nx = (ndim >= 1) ? hdr.dim[1] : 1;
            _ny = (ndim >= 2) ? hdr.dim[2] : 1;
            _nz = (ndim >= 3) ? hdr.dim[3] : 1;
            _nt = (ndim >= 4 && hdr.dim[4] > 0) ? hdr.dim[4] : 1;
            _datatype  = hdr.datatype;
            _vox_offset = hdr.vox_offset;
            _scl_slope = hdr.scl_slope;
            _scl_inter = hdr.scl_inter;
            for (int i = 0; i < 8; ++i) _pixdim[i] = hdr.pixdim[i];
        } else {
            // NIFTI-2
            if (_raw_data.size() < 540) {
                throw std::runtime_error("NiftiReaderCPP: compressed file too small for NIFTI-2 header");
            }
            Nifti2Header hdr;
            std::memcpy(&hdr, _raw_data.data(), sizeof(hdr));

            if (_byte_swapped) {
                for (int i = 0; i < 8; ++i) bswap8(&hdr.dim[i]);
                bswap2(&hdr.datatype);
                for (int i = 0; i < 8; ++i) bswap8(&hdr.pixdim[i]);
                bswap8(&hdr.vox_offset);
                bswap8(&hdr.scl_slope);
                bswap8(&hdr.scl_inter);
            }

            int64_t ndim = hdr.dim[0];
            _nx = (ndim >= 1) ? hdr.dim[1] : 1;
            _ny = (ndim >= 2) ? hdr.dim[2] : 1;
            _nz = (ndim >= 3) ? hdr.dim[3] : 1;
            _nt = (ndim >= 4 && hdr.dim[4] > 0) ? hdr.dim[4] : 1;
            _datatype   = hdr.datatype;
            _vox_offset = static_cast<double>(hdr.vox_offset);
            _scl_slope  = static_cast<float>(hdr.scl_slope);
            _scl_inter  = static_cast<float>(hdr.scl_inter);
            for (int i = 0; i < 8; ++i) _pixdim[i] = static_cast<float>(hdr.pixdim[i]);
        }
    } else {
        // Plain .nii — read directly from file
        std::ifstream f(_fname, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("NiftiReaderCPP: cannot open file: " + _fname);
        }

        int32_t sh = 0;
        f.read(reinterpret_cast<char*>(&sh), sizeof(sh));
        if (!f) throw std::runtime_error("NiftiReaderCPP: cannot read sizeof_hdr");

        bool swapped = false;
        int32_t sh_swapped = sh;
        bswap4(&sh_swapped);

        if (sh == 348 || sh == 540) {
            swapped = false;
        } else if (sh_swapped == 348 || sh_swapped == 540) {
            swapped = true;
            sh = sh_swapped;
        } else {
            throw std::runtime_error("NiftiReaderCPP: invalid sizeof_hdr");
        }
        _byte_swapped = swapped;

        f.seekg(0, std::ios::beg);

        if (sh == 348) {
            Nifti1Header hdr;
            f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            if (!f) throw std::runtime_error("NiftiReaderCPP: cannot read NIFTI-1 header");

            if (_byte_swapped) {
                for (int i = 0; i < 8; ++i) bswap2(&hdr.dim[i]);
                bswap2(&hdr.datatype);
                for (int i = 0; i < 8; ++i) bswap4(&hdr.pixdim[i]);
                bswap4(&hdr.vox_offset);
                bswap4(&hdr.scl_slope);
                bswap4(&hdr.scl_inter);
            }

            int ndim = hdr.dim[0];
            _nx = (ndim >= 1) ? hdr.dim[1] : 1;
            _ny = (ndim >= 2) ? hdr.dim[2] : 1;
            _nz = (ndim >= 3) ? hdr.dim[3] : 1;
            _nt = (ndim >= 4 && hdr.dim[4] > 0) ? hdr.dim[4] : 1;
            _datatype   = hdr.datatype;
            _vox_offset = hdr.vox_offset;
            _scl_slope  = hdr.scl_slope;
            _scl_inter  = hdr.scl_inter;
            for (int i = 0; i < 8; ++i) _pixdim[i] = hdr.pixdim[i];
        } else {
            // NIFTI-2
            Nifti2Header hdr;
            f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            if (!f) throw std::runtime_error("NiftiReaderCPP: cannot read NIFTI-2 header");

            if (_byte_swapped) {
                for (int i = 0; i < 8; ++i) bswap8(&hdr.dim[i]);
                bswap2(&hdr.datatype);
                for (int i = 0; i < 8; ++i) bswap8(&hdr.pixdim[i]);
                bswap8(&hdr.vox_offset);
                bswap8(&hdr.scl_slope);
                bswap8(&hdr.scl_inter);
            }

            int64_t ndim = hdr.dim[0];
            _nx = (ndim >= 1) ? hdr.dim[1] : 1;
            _ny = (ndim >= 2) ? hdr.dim[2] : 1;
            _nz = (ndim >= 3) ? hdr.dim[3] : 1;
            _nt = (ndim >= 4 && hdr.dim[4] > 0) ? hdr.dim[4] : 1;
            _datatype   = hdr.datatype;
            _vox_offset = static_cast<double>(hdr.vox_offset);
            _scl_slope  = static_cast<float>(hdr.scl_slope);
            _scl_inter  = static_cast<float>(hdr.scl_inter);
            for (int i = 0; i < 8; ++i) _pixdim[i] = static_cast<float>(hdr.pixdim[i]);
        }
    }

    // Validate datatype
    _data_type = nifti_dtype_to_string(_datatype);
    if (_data_type == "unknown") {
        throw std::runtime_error("NiftiReaderCPP: unsupported NIFTI datatype code: " +
                                 std::to_string(_datatype));
    }

    // Map to bfiocpp convention: X=width, Y=height, Z=depth, T=tsteps, C=channels
    _image_width   = _nx;
    _image_height  = _ny;
    _image_depth   = (_nz > 0) ? _nz : 1;
    _num_tsteps    = (_nt > 0) ? _nt : 1;
    _num_channels  = 1;

    // If scl_slope is non-zero, output will be float64
    if (_scl_slope != 0.0f) {
        _data_type = "double";
    }
}

// ---------------------------------------------------------------------------
// EnsureDataLoaded — inflate .nii.gz (called once via std::once_flag)
// ---------------------------------------------------------------------------
void NiftiReaderCPP::EnsureDataLoaded() {
    std::call_once(_load_flag, [this]() {
        std::ifstream f(_fname, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            throw std::runtime_error("NiftiReaderCPP: cannot open .nii.gz file: " + _fname);
        }
        auto compressed_size = static_cast<std::size_t>(f.tellg());
        f.seekg(0, std::ios::beg);

        std::vector<uint8_t> compressed(compressed_size);
        f.read(reinterpret_cast<char*>(compressed.data()), compressed_size);
        f.close();

        // Inflate using zlib with gzip support (windowBits = 15+16)
        z_stream stream{};
        stream.next_in   = compressed.data();
        stream.avail_in  = static_cast<uInt>(compressed_size);

        if (inflateInit2(&stream, 16 + 15) != Z_OK) {
            throw std::runtime_error("NiftiReaderCPP: inflateInit2 failed");
        }

        const std::size_t CHUNK = 1024 * 1024; // 1 MB chunks
        _raw_data.clear();

        int ret;
        do {
            std::size_t old_size = _raw_data.size();
            _raw_data.resize(old_size + CHUNK);
            stream.next_out  = _raw_data.data() + old_size;
            stream.avail_out = static_cast<uInt>(CHUNK);
            ret = inflate(&stream, Z_NO_FLUSH);
            if (ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
                inflateEnd(&stream);
                throw std::runtime_error("NiftiReaderCPP: inflate error code=" + std::to_string(ret));
            }
            // Trim the unused portion of the last chunk
            _raw_data.resize(_raw_data.size() - stream.avail_out);
        } while (ret == Z_OK);  // Z_STREAM_END exits; Z_BUF_ERROR is benign

        inflateEnd(&stream);
    });
}

// ---------------------------------------------------------------------------
// Byte-swap a single value of type T
// ---------------------------------------------------------------------------
template<typename T>
static T maybe_swap(T val, bool swap) {
    if (!swap) return val;
    uint8_t buf[sizeof(T)];
    std::memcpy(buf, &val, sizeof(T));
    std::reverse(buf, buf + sizeof(T));
    std::memcpy(&val, buf, sizeof(T));
    return val;
}

// ---------------------------------------------------------------------------
// ReadRegion<T> — reads a rectangular sub-region without scaling
// ---------------------------------------------------------------------------
template<typename T>
std::shared_ptr<image_data> NiftiReaderCPP::ReadRegion(
    const Seq& rows, const Seq& cols, const Seq& layers,
    const Seq& channels, const Seq& tsteps)
{
    // Clamp to valid ranges (0-indexed, inclusive)
    auto y0 = std::max<long>(rows.Start(), 0);
    auto y1 = std::min<long>(rows.Stop(),  _image_height - 1);
    auto x0 = std::max<long>(cols.Start(), 0);
    auto x1 = std::min<long>(cols.Stop(),  _image_width - 1);
    auto z0 = std::max<long>(layers.Start(), 0);
    auto z1 = std::min<long>(layers.Stop(),  _image_depth - 1);
    auto t0 = std::max<long>(tsteps.Start(), 0);
    auto t1 = std::min<long>(tsteps.Stop(),  _num_tsteps - 1);

    std::size_t ny = static_cast<std::size_t>(y1 - y0 + 1);
    std::size_t nx = static_cast<std::size_t>(x1 - x0 + 1);
    std::size_t nz = static_cast<std::size_t>(z1 - z0 + 1);
    std::size_t nt = static_cast<std::size_t>(t1 - t0 + 1);

    std::vector<T> out;
    out.reserve(nt * 1 * nz * ny * nx);

    // vox_offset is where pixel data starts in the file
    std::size_t vox_start = static_cast<std::size_t>(_vox_offset);
    std::size_t elem_size = sizeof(T);

    if (_is_gz) {
        EnsureDataLoaded();

        for (long it = t0; it <= t1; ++it) {
            for (long iz = z0; iz <= z1; ++iz) {
                for (long iy = y0; iy <= y1; ++iy) {
                    for (long ix = x0; ix <= x1; ++ix) {
                        // NIFTI flat index: x-fastest
                        std::size_t flat = static_cast<std::size_t>(
                            ix + _nx * (iy + _ny * (iz + _nz * it)));
                        std::size_t byte_off = vox_start + flat * elem_size;
                        if (byte_off + elem_size > _raw_data.size()) {
                            throw std::runtime_error("NiftiReaderCPP: out-of-bounds read in compressed data");
                        }
                        T val;
                        std::memcpy(&val, _raw_data.data() + byte_off, elem_size);
                        out.push_back(maybe_swap(val, _byte_swapped));
                    }
                }
            }
        }
    } else {
        std::ifstream f(_fname, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("NiftiReaderCPP: cannot open file: " + _fname);
        }

        for (long it = t0; it <= t1; ++it) {
            for (long iz = z0; iz <= z1; ++iz) {
                for (long iy = y0; iy <= y1; ++iy) {
                    for (long ix = x0; ix <= x1; ++ix) {
                        std::size_t flat = static_cast<std::size_t>(
                            ix + _nx * (iy + _ny * (iz + _nz * it)));
                        std::size_t byte_off = vox_start + flat * elem_size;
                        f.seekg(static_cast<std::streamoff>(byte_off), std::ios::beg);
                        T val;
                        f.read(reinterpret_cast<char*>(&val), elem_size);
                        if (!f) {
                            throw std::runtime_error("NiftiReaderCPP: read error at offset " +
                                                     std::to_string(byte_off));
                        }
                        out.push_back(maybe_swap(val, _byte_swapped));
                    }
                }
            }
        }
    }

    return std::make_shared<image_data>(std::move(out));
}

// ---------------------------------------------------------------------------
// ReadRegionScaled — reads and applies scl_slope / scl_inter, returns double
// ---------------------------------------------------------------------------
std::shared_ptr<image_data> NiftiReaderCPP::ReadRegionScaled(
    const Seq& rows, const Seq& cols, const Seq& layers,
    const Seq& channels, const Seq& tsteps)
{
    // Read raw data into a temporary image_data, then convert to double + scale
    std::shared_ptr<image_data> raw;
    switch (_datatype) {
        case 2:    raw = ReadRegion<uint8_t> (rows, cols, layers, channels, tsteps); break;
        case 4:    raw = ReadRegion<int16_t> (rows, cols, layers, channels, tsteps); break;
        case 8:    raw = ReadRegion<int32_t> (rows, cols, layers, channels, tsteps); break;
        case 16:   raw = ReadRegion<float>   (rows, cols, layers, channels, tsteps); break;
        case 64:   raw = ReadRegion<double>  (rows, cols, layers, channels, tsteps); break;
        case 256:  raw = ReadRegion<int8_t>  (rows, cols, layers, channels, tsteps); break;
        case 512:  raw = ReadRegion<uint16_t>(rows, cols, layers, channels, tsteps); break;
        case 768:  raw = ReadRegion<uint32_t>(rows, cols, layers, channels, tsteps); break;
        case 1024: raw = ReadRegion<int64_t> (rows, cols, layers, channels, tsteps); break;
        case 1280: raw = ReadRegion<uint64_t>(rows, cols, layers, channels, tsteps); break;
        default:
            throw std::runtime_error("NiftiReaderCPP: unsupported datatype for scaling");
    }

    double slope = static_cast<double>(_scl_slope);
    double inter = static_cast<double>(_scl_inter);

    // Convert raw variant → std::vector<double> with scaling applied
    std::vector<double> out;
    std::visit([&](auto&& vec) {
        out.reserve(vec.size());
        for (auto v : vec) {
            out.push_back(static_cast<double>(v) * slope + inter);
        }
    }, *raw);

    return std::make_shared<image_data>(std::move(out));
}

// ---------------------------------------------------------------------------
// GetImageData — public dispatch
// ---------------------------------------------------------------------------
std::shared_ptr<image_data> NiftiReaderCPP::GetImageData(
    const Seq& rows, const Seq& cols,
    const Seq& layers, const Seq& channels, const Seq& tsteps)
{
    if (_scl_slope != 0.0f) {
        return ReadRegionScaled(rows, cols, layers, channels, tsteps);
    }

    switch (_datatype) {
        case 2:    return ReadRegion<uint8_t> (rows, cols, layers, channels, tsteps);
        case 4:    return ReadRegion<int16_t> (rows, cols, layers, channels, tsteps);
        case 8:    return ReadRegion<int32_t> (rows, cols, layers, channels, tsteps);
        case 16:   return ReadRegion<float>   (rows, cols, layers, channels, tsteps);
        case 64:   return ReadRegion<double>  (rows, cols, layers, channels, tsteps);
        case 256:  return ReadRegion<int8_t>  (rows, cols, layers, channels, tsteps);
        case 512:  return ReadRegion<uint16_t>(rows, cols, layers, channels, tsteps);
        case 768:  return ReadRegion<uint32_t>(rows, cols, layers, channels, tsteps);
        case 1024: return ReadRegion<int64_t> (rows, cols, layers, channels, tsteps);
        case 1280: return ReadRegion<uint64_t>(rows, cols, layers, channels, tsteps);
        default:
            throw std::runtime_error("NiftiReaderCPP: unsupported datatype: " +
                                     std::to_string(_datatype));
    }
}

} // namespace bfiocpp
