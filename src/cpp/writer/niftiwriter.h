#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <pybind11/numpy.h>
#include "../utilities/sequence.h"

namespace py = pybind11;

namespace bfiocpp {

class NiftiWriterCPP {
public:
    NiftiWriterCPP(
        const std::string& fname,
        const std::vector<std::int64_t>& image_shape,
        const std::string& dtype_str,
        const std::string& dimension_order
    );
    ~NiftiWriterCPP();

    void WriteImageData(
        const py::array& py_image,
        const Seq& rows,
        const Seq& cols,
        const std::optional<Seq>& layers,
        const std::optional<Seq>& channels,
        const std::optional<Seq>& tsteps
    );

    void Close();  // explicit flush for .nii.gz; safe to call multiple times

private:
    void WriteNifti1Header(std::ostream& out);
    void FlushGzip();

    std::string _fname;
    bool _is_gz;
    bool _closed;

    // Image dimensions (NIFTI order: X=width, Y=height, Z=depth, T=tsteps)
    int64_t _nx, _ny, _nz, _nt;
    int16_t _nifti_datatype;
    int     _elem_size;    // bytes per voxel
    double  _vox_offset;   // 352.0 for NIFTI-1 single-file

    // Index of each dimension in the caller's image_shape / Seq arguments
    int _x_index, _y_index;
    std::optional<int> _z_index, _c_index, _t_index;

    // Pixel buffer for .nii.gz (full image; flushed on Close())
    std::vector<uint8_t> _pixel_buf;
};

} // namespace bfiocpp
