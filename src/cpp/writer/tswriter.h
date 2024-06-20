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

/*
struct VisitImage {
  void operator()(const std::vector<std::uint8_t>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }
  void operator()(const std::vector<std::uint16_t>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }
  void operator()(const std::vector<std::uint32_t>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }
  void operator()(const std::vector<std::uint64_t>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }

  void operator()(const std::vector<std::int8_t>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }
  void operator()(const std::vector<std::int16_t>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }
  void operator()(const std::vector<std::int32_t>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }
  void operator()(const std::vector<std::int64_t>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }
  void operator()(const std::vector<float>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }
  void operator()(const std::vector<double>& image_vec, 
                  const std::vector<std::int64_t>& image_shape, 
                  const std::vector<std::int64_t>& chunk_shape,  
                  const std::string& filename, 
                  const std::string& dtype_str) {

    auto data_array = tensorstore::Array(image_vec.data(), image_shape, tensorstore::c_order);

    auto spec = GetZarrSpecToWrite(filename, image_shape, chunk_shape, dtype_str);

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                              spec,
                              tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                              tensorstore::ReadWriteMode::write).result());


    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  }
};
*/


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
    std::uint16_t _data_type_code;
    std::optional<int>_z_index, _c_index, _t_index;        

    std::string get_variant_type(const image_data& image) {
        return std::visit([](auto&& arg) -> std::string {
            using T = std::decay_t<decltype(arg)>;

            if constexpr (std::is_same_v<T, std::vector<std::uint8_t>>) return "uint8";
            else if constexpr (std::is_same_v<T, std::vector<std::uint16_t>>) return "uint16";
            else if constexpr (std::is_same_v<T, std::vector<std::uint32_t>>) return "uint32";
            else if constexpr (std::is_same_v<T, std::vector<std::uint64_t>>) return "uint64";
            else if constexpr (std::is_same_v<T, std::vector<std::int8_t>>) return "int8";
            else if constexpr (std::is_same_v<T, std::vector<std::int16_t>>) return "int16";
            else if constexpr (std::is_same_v<T, std::vector<std::int32_t>>) return "int32";
            else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) return "int64";
            else if constexpr (std::is_same_v<T, std::vector<float>>) return "float";
            else if constexpr (std::is_same_v<T, std::vector<double>>) return "double";
            else return "Unknown type";
        }, image);
    }
};
}

