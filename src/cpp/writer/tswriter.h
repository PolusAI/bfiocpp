#pragma once

#include <string>
#include <vector>
#include <optional>
#include "tensorstore/tensorstore.h"
#include "../utilities/sequence.h"
#include "../utilities/utilities.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace bfiocpp{

class TsWriterCPP{
public:
    TsWriterCPP (
        const std::string& fname,
        const std::vector<std::int64_t>& image_shape,
        const std::vector<std::int64_t>& chunk_shape,
        const std::string& dtype_str,
        const std::string& dimension_order,
        FileType file_type = FileType::OmeZarrV2
    );

    void WriteImageData (
        const py::array& py_image, 
        const Seq& rows, 
        const Seq& cols, 
        const std::optional<Seq>& layers, 
        const std::optional<Seq>& channels, 
        const std::optional<Seq>& tsteps
    );

private:
    std::string _filename;

    std::vector<std::int64_t> _image_shape, _chunk_shape;
    
    uint16_t _dtype_code;

    tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic> _source;

    std::optional<int>_z_index, _c_index, _t_index;
    int _x_index, _y_index;

};
}

