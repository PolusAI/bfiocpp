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
    const std::string& dtype_str,
    const std::string& dimension_order
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

    auto position = dimension_order.find("X");
    if (position != std::string::npos) _x_index = position;

    position = dimension_order.find("Y");
    if (position != std::string::npos) _y_index = position;

    position = dimension_order.find("C");
    if (position != std::string::npos) _c_index.emplace(position);

    position = dimension_order.find("T");
    if (position != std::string::npos) _t_index.emplace(position);

    position = dimension_order.find("Z");
    if (position != std::string::npos) _z_index.emplace(position);
}

void TsWriterCPP::WriteImageData(
    py::array& py_image,
    const Seq& rows,
    const Seq& cols,
    const std::optional<Seq>& layers,
    const std::optional<Seq>& channels,
    const std::optional<Seq>& tsteps) {

    std::vector<std::int64_t> shape;

    auto output_transform = tensorstore::IdentityTransform(_source.domain());

    if (_z_index.has_value() && layers.has_value()) {
        output_transform = (std::move(output_transform) | tensorstore::Dims(_z_index.value()).ClosedInterval(layers.value().Start(), layers.value().Stop())).value();
        shape.emplace_back(layers.value().Stop() - layers.value().Start()+1);
    }

    if (_c_index.has_value() && channels.has_value()) {
        output_transform = (std::move(output_transform) | tensorstore::Dims(_c_index.value()).ClosedInterval(channels.value().Start(), channels.value().Stop())).value();
        shape.emplace_back(channels.value().Stop() - channels.value().Start()+1);
    }

    if (_t_index.has_value() && tsteps.has_value()) {
        output_transform = (std::move(output_transform) | tensorstore::Dims(_t_index.value()).ClosedInterval(tsteps.value().Start(), tsteps.value().Stop())).value();
        shape.emplace_back(tsteps.value().Stop() - tsteps.value().Start()+1);
    }

    output_transform = (std::move(output_transform) | tensorstore::Dims(_y_index).ClosedInterval(rows.Start(), rows.Stop()) |
                                                      tensorstore::Dims(_x_index).ClosedInterval(cols.Start(), cols.Stop())).value();



    shape.emplace_back(rows.Stop() - rows.Start()+1);
    shape.emplace_back(cols.Stop() - cols.Start()+1);

    // use switch instead of template to avoid creating functions for each datatype
    switch(_dtype_code)
    {
        case (1): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint8_t, 1>().data(0), shape, tensorstore::c_order);

            // Write data array to TensorStore
            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();

            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }

            break;
        }
        case (2): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint16_t, 1>().data(0), shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (4): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint32_t, 1>().data(0), shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (8): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::uint64_t, 1>().data(0), shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (16): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int8_t, 1>().data(0), shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (32): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int16_t, 1>().data(0), shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (64): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int32_t, 1>().data(0), shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (128): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<std::int64_t, 1>().data(0), shape, tensorstore::c_order);

            // Write data array to TensorStore
            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (256): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<float, 1>().data(0), shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();
            if (!write_result.ok()) {
                std::cerr << "Error writing image: " << write_result.status() << std::endl;
            }
            break;
        }
        case (512): {
            auto data_array = tensorstore::Array(py_image.mutable_unchecked<double, 1>().data(0), shape, tensorstore::c_order);

            auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data_array), _source | output_transform).result();
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

} // end ns bfiocpp
