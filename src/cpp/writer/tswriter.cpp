#include "tswriter.h"

#include "../utilities/utilities.h"

#include <variant>
#include <string>

using ::tensorstore::internal_zarr::ChooseBaseDType;

namespace bfiocpp {

TsWriterCPP::TsWriterCPP(
    const std::string& fname, 
    const std::vector<std::int64_t>& image_shape, 
    const std::vector<std::int64_t>& chunk_shape,
    const std::string& dtype_str
  ): _filename(fname), _image_shape(image_shape), _chunk_shape(chunk_shape) {
  
  _dtype_code = GetDataTypeCode(dtype_str); 

  auto spec = GetZarrSpecToWrite(_filename, image_shape, chunk_shape, dtype_str);

  TENSORSTORE_CHECK_OK_AND_ASSIGN(_source, tensorstore::Open(
                            spec,
                            tensorstore::OpenMode::create |
                            tensorstore::OpenMode::delete_existing,
                            tensorstore::ReadWriteMode::write).result());
}


void TsWriterCPP::WriteImage(py::array& py_image) {

  // use switch instead of template to avoid creating functions for each datatype
  switch(_dtype_code)
  {
    case (1): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint8_t, 1>().data(0), _image_shape, tensorstore::c_order);

        // Write data array to TensorStore
        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();

        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }

        break;
    }
    case (2): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint16_t, 1>().data(0), _image_shape, tensorstore::c_order);

        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();
        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }
        break;  
    }  
    case (4): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint32_t, 1>().data(0), _image_shape, tensorstore::c_order);

        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();
        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }
        break;
    }
    case (8): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint64_t, 1>().data(0), _image_shape, tensorstore::c_order);

        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();
        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }
        break;
    }
    case (16): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int8_t, 1>().data(0), _image_shape, tensorstore::c_order);

        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();
        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }
        break;
    }
    case (32): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int16_t, 1>().data(0), _image_shape, tensorstore::c_order);

        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();
        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }
        break;   
    } 
    case (64): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int32_t, 1>().data(0), _image_shape, tensorstore::c_order);

        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();
        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }
        break;
    }
    case (128): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int64_t, 1>().data(0), _image_shape, tensorstore::c_order);

        // Write data array to TensorStore
        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();
        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }
        break;
    }
    case (256): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<float, 1>().data(0), _image_shape, tensorstore::c_order);

        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();
        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }
        break;
    }
    case (512): {
        auto data_array = tensorstore::Array(py_image.mutable_unchecked<double, 1>().data(0), _image_shape, tensorstore::c_order);

        auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source).result();
        if (!write_result.ok()) {
            std::cerr << "Error writing image: " << write_result.status() << std::endl;
        }
        break;
    }
    default: {
        // should not be reached
        std::cerr << "Error writing image: unsupported data type" << std::endl;
    }
  }
  }
} 
