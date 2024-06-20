#include "tswriter.h"

#include "../utilities/utilities.h"

#include <variant>
#include <string>

using ::tensorstore::internal_zarr::ChooseBaseDType;

namespace bfiocpp {

TsWriterCPP::TsWriterCPP(const std::string& fname): _filename(fname) {

}

void TsWriterCPP::SetImageHeight(std::int64_t image_height) {_image_height = image_height;}
void TsWriterCPP::SetImageWidth (std::int64_t image_width) {_image_width = image_width;}
void TsWriterCPP::SetImageDepth (std::int64_t image_depth) {_image_depth = image_depth;}
void TsWriterCPP::SetTileHeight (std::int64_t tile_height) {_tile_height = tile_height;}
void TsWriterCPP::SetTileWidth (std::int64_t tile_width) {_tile_width = tile_width;}
void TsWriterCPP::SetTileDepth (std::int64_t tile_depth) {_tile_depth = tile_depth;}
void TsWriterCPP::SetChannelCount (std::int64_t num_channels) {_num_channels = num_channels;}
void TsWriterCPP::SetTstepCount (std::int64_t num_tsteps) {_num_tsteps = num_tsteps;}
void TsWriterCPP::SetDataType(const std::string& data_type) {_data_type = data_type;}

void TsWriterCPP::write_image(py::array& py_image, const std::vector<std::int64_t>& image_shape, const std::vector<std::int64_t>& chunk_shape) {

  auto dt = py_image.dtype();
  auto dt_str = py::str(dt).cast<std::string>();

  dt_str = (dt_str == "float64") ? "double" : dt_str; // change float64 numpy type to double

  auto dtype = GetTensorStoreDataType(dt_str);

  std::string dtype_str = ChooseBaseDType(dtype).value().encoded_dtype;

  auto spec = GetZarrSpecToWrite(_filename, image_shape, chunk_shape, dtype_str);

  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                            spec,
                            tensorstore::OpenMode::create |
                            tensorstore::OpenMode::delete_existing,
                            tensorstore::ReadWriteMode::write).result());

  // use if-else statement instead of template to avoid creating functions for each datatype
  if (dt_str == "uint8") {
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint8_t, 1>().data(0), image_shape, tensorstore::c_order);

    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  } else if (dt_str == "uint16") {
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint16_t, 1>().data(0), image_shape, tensorstore::c_order);

    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }

  } else if (dt_str == "uint32") {
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint32_t, 1>().data(0), image_shape, tensorstore::c_order);

    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  } else if (dt_str == "uint64") {
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint64_t, 1>().data(0), image_shape, tensorstore::c_order);

    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  } else if (dt_str == "int8") {
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int8_t, 1>().data(0), image_shape, tensorstore::c_order);

    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }

  } else if (dt_str == "int16") {
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int16_t, 1>().data(0), image_shape, tensorstore::c_order);

    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  } else if (dt_str == "int32") {
  
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int32_t, 1>().data(0), image_shape, tensorstore::c_order);

    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  } else if (dt_str == "int64") {
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int64_t, 1>().data(0), image_shape, tensorstore::c_order);

    // Write data array to TensorStore
    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  } else if (dt_str == "float") {
    
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<float, 1>().data(0), image_shape, tensorstore::c_order);

    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  } else if (dt_str == "double") {
    
    auto data_array = tensorstore::Array(py_image.mutable_unchecked<double, 1>().data(0), image_shape, tensorstore::c_order);

    auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), store).result();
    if (!write_result.ok()) {
        std::cerr << "Error writing image: " << write_result.status() << std::endl;
    }
  } else { 
      throw std::runtime_error("Unsupported data type: " + dt_str);
  }                    
  }
} 