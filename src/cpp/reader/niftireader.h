#pragma once

#include <string>
#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>
#include "../utilities/sequence.h"
#include "../reader/tsreader.h"  // for image_data

namespace bfiocpp {

class NiftiReaderCPP {
public:
    explicit NiftiReaderCPP(const std::string& fname);

    std::int64_t GetImageHeight()   const { return _image_height; }
    std::int64_t GetImageWidth()    const { return _image_width;  }
    std::int64_t GetImageDepth()    const { return _image_depth;  }
    std::int64_t GetChannelCount()  const { return _num_channels; }
    std::int64_t GetTstepCount()    const { return _num_tsteps;   }
    // Tile dims = full image dims (no tiling for NIFTI)
    std::int64_t GetTileHeight()    const { return _image_height; }
    std::int64_t GetTileWidth()     const { return _image_width;  }
    std::int64_t GetTileDepth()     const { return _image_depth;  }

    std::string  GetDataType() const { return _data_type; }

    float GetPhysicalSizeX() const { return _pixdim[1]; }
    float GetPhysicalSizeY() const { return _pixdim[2]; }
    float GetPhysicalSizeZ() const { return _pixdim[3]; }

    std::shared_ptr<image_data> GetImageData(
        const Seq& rows, const Seq& cols,
        const Seq& layers, const Seq& channels, const Seq& tsteps);

private:
    void ReadHeader();
    void EnsureDataLoaded();  // inflate .nii.gz → _raw_data (thread-safe)

    template<typename T>
    std::shared_ptr<image_data> ReadRegion(
        const Seq& rows, const Seq& cols, const Seq& layers,
        const Seq& channels, const Seq& tsteps);

    std::shared_ptr<image_data> ReadRegionScaled(
        const Seq& rows, const Seq& cols, const Seq& layers,
        const Seq& channels, const Seq& tsteps);

    std::string  _fname;
    bool         _is_gz       = false;
    bool         _byte_swapped = false;

    // NIFTI dimension (stored as int64 to cover both NIFTI-1 and NIFTI-2)
    int64_t _nx = 1, _ny = 1, _nz = 1, _nt = 1;
    int16_t _datatype = 0;     // raw NIFTI datatype code
    double  _vox_offset = 0.0;
    float   _scl_slope  = 0.0f;
    float   _scl_inter  = 0.0f;
    float   _pixdim[8]  = {};

    // bfiocpp-style dimensions
    std::int64_t _image_width   = 1;
    std::int64_t _image_height  = 1;
    std::int64_t _image_depth   = 1;
    std::int64_t _num_channels  = 1;
    std::int64_t _num_tsteps    = 1;
    std::string  _data_type;

    // Compressed data cache
    std::vector<uint8_t> _raw_data;
    std::once_flag       _load_flag;
};

} // namespace bfiocpp
