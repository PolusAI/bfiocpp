#pragma once

#include <string>
#include <memory>
#include <vector>
#include <variant>
#include <iostream>
#include <tuple>
#include <optional>
#include <unordered_map>
#include "../reader/sequence.h"

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace bfiocpp{

class TsWriterCPP{
public:
    TsWriterCPP(const std::string& fname, const std::vector<std::int64_t>& image_shape, const std::vector<std::int64_t>& chunk_shape, const std::string& dtype);

    void write_image(py::array& py_image);

private:
    std::string _filename;

    std::vector<std::int64_t> _image_shape, _chunk_shape;
    
    uint16_t _dtype_code;

    tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic> _source;

};
}

