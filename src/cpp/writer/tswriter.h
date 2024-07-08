#pragma once

#include <string>
#include <vector>
#include "tensorstore/tensorstore.h"
#include "../utilities/sequence.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace bfiocpp{

class TsWriterCPP{
public:
    TsWriterCPP(
        const std::string& fname, 
        const std::vector<std::int64_t>& image_shape, 
        const std::vector<std::int64_t>& chunk_shape, 
        const std::string& dtype
    );

    void WriteImageData(
        py::array& py_image, 
        const Seq& rows, 
        const Seq& cols, 
        const Seq& layers, 
        const Seq& channels, 
        const Seq& tsteps
    );

private:
    std::string _filename;

    std::vector<std::int64_t> _image_shape, _chunk_shape;
    
    uint16_t _dtype_code;

    tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic> _source;

    tensorstore::IndexTransform<> output_transform_;

    void SetOutputTransform(
        const Seq& rows, 
        const Seq& cols, 
        const Seq& layers, 
        const Seq& channels, 
        const Seq& tsteps
    );

};
}

