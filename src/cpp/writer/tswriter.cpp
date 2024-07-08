#include <string>

#include "tensorstore/array.h"
#include "tensorstore/open.h"
#include "tensorstore/index_space/dim_expression.h"

#include "tswriter.h"
#include "../utilities/utilities.h"


namespace bfiocpp {

TsWriterCPP::TsWriterCPP(
    const std::string& fname, 
    const std::vector<std::int64_t>& image_shape, 
    const std::vector<std::int64_t>& chunk_shape,
    const std::string& dtype_str
  ): _filename(fname), 
     _image_shape(image_shape), 
     _chunk_shape(chunk_shape),  
     _dtype_code(GetDataTypeCode(dtype_str)) {


    TENSORSTORE_CHECK_OK_AND_ASSIGN(_source, tensorstore::Open(
        GetZarrSpecToWrite(_filename, _image_shape, _chunk_shape, GetEncodedType(_dtype_code)),
        tensorstore::OpenMode::create |
        tensorstore::OpenMode::delete_existing,
        tensorstore::ReadWriteMode::write).result()
    );
}


void TsWriterCPP::WriteImageData(
    py::array& py_image, 
    const Seq& rows, 
    const Seq& cols, 
    const Seq& layers, 
    const Seq& channels, 
    const Seq& tsteps) {

    SetOutputTransform(rows, cols, layers, channels, tsteps);

    // use switch instead of template to avoid creating functions for each datatype
    switch(_dtype_code)
    {
        case (1): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint8_t, 1>().data(0), _image_shape, tensorstore::c_order);

            // Write data array to TensorStore
            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();

            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }

            break;
        }
        case (2): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint16_t, 1>().data(0), _image_shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;  
        }  
        case (4): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint32_t, 1>().data(0), _image_shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (8): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint64_t, 1>().data(0), _image_shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (16): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int8_t, 1>().data(0), _image_shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (32): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int16_t, 1>().data(0), _image_shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;   
        } 
        case (64): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int32_t, 1>().data(0), _image_shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (128): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int64_t, 1>().data(0), _image_shape, tensorstore::c_order);

            // Write data array to TensorStore
            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (256): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<float, 1>().data(0), _image_shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (512): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<double, 1>().data(0), _image_shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform_).result();
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

void TsWriterCPP::SetOutputTransform(
        const Seq& rows, 
        const Seq& cols, 
        const Seq& layers, 
        const Seq& channels, 
        const Seq& tsteps
) {

    output_transform_ = tensorstore::IdentityTransform(_source.domain());

    int z_index=0,
        c_index=1,         t_index=2,
        y_index=3,
        x_index=4;

    if (layers.IsValidDimension()) {
        output_transform_ = (std::move(output_transform_) | tensorstore::Dims(z_index).ClosedInterval(layers.Start(), layers.Stop())).value();
    } else {
        --c_index;
        --t_index;
        --y_index;
        --x_index;
    }

    if (channels.IsValidDimension()) {
        output_transform_ = (std::move(output_transform_) | tensorstore::Dims(c_index).ClosedInterval(channels.Start(), channels.Stop())).value();
    } else {
        --t_index;
        --y_index;
        --x_index;
    }

    if (tsteps.IsValidDimension()) {
        output_transform_ = (std::move(output_transform_) | tensorstore::Dims(t_index).ClosedInterval(tsteps.Start(), tsteps.Stop())).value();
    } else {
        --y_index;
        --x_index;
    }

    output_transform_ = (std::move(output_transform_) | tensorstore::Dims(y_index).ClosedInterval(rows.Start(), rows.Stop()) |
                                                        tensorstore::Dims(x_index).ClosedInterval(cols.Start(), cols.Stop())).value(); 
}


} // end ns bfiocpp
