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


using image_data = std::variant<std::vector<std::uint8_t>,
                                std::vector<std::uint16_t>, 
                                std::vector<std::uint32_t>, 
                                std::vector<std::uint64_t>, 
                                std::vector<std::int8_t>, 
                                std::vector<std::int16_t>,
                                std::vector<std::int32_t>,
                                std::vector<std::int64_t>,
                                std::vector<float>,
                                std::vector<double>>;



using iter_indicies = std::tuple<std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t>;

namespace bfiocpp{

class TsWriterCPP{
public:
    TsWriterCPP(const std::string& fname);

    void SetImageHeight(std::int64_t image_height);
    void SetImageWidth (std::int64_t image_width);
    void SetImageDepth (std::int64_t image_depth);
    void SetTileHeight (std::int64_t tile_height);
    void SetTileWidth (std::int64_t tile_width);
    void SetTileDepth (std::int64_t tile_depth);
    void SetChannelCount (std::int64_t num_channels);
    void SetTstepCount (std::int64_t num_steps);
    void SetDataType(const std::string& data_type);

    void write_image(py::array& py_image, const std::vector<std::int64_t>& image_shape, const std::vector<std::int64_t>& chunk_shape);

private:
    std::string _filename, _data_type;
    std::int64_t    _image_height, 
                    _image_width, 
                    _image_depth, 
                    _tile_height, 
                    _tile_width, 
                    _tile_depth, 
                    _num_channels,
                    _num_tsteps;
};
}

