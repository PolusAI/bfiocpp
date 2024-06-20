#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <tuple>
#include "../reader/tsreader.h"
#include "../reader/sequence.h"
#include "../utilities/utilities.h"
#include "../writer/tswriter.h"

namespace py = pybind11;
using bfiocpp::Seq;

#define EXTRACT_FROM_VARIANT_AND_RETURN(N) \
    case N: { \
        auto size = std::get<N>(*seq_ptr).size(); \
        auto data = std::get<N>(*seq_ptr).data(); \
        auto capsule = py::capsule(new auto (seq_ptr), [](void *p) {delete reinterpret_cast<decltype(seq_ptr)*>(p);}); \
        return py::array(size, data, capsule).reshape({num_tsteps, num_channels, num_layers, num_rows, num_cols}); \
        break; \
    }

inline py::array as_pyarray_shared_5d(std::shared_ptr<image_data> seq_ptr, size_t num_rows, size_t num_cols, size_t num_layers=1, size_t num_channels=1, size_t num_tsteps=1 ) {
    switch (seq_ptr->index()) {
        EXTRACT_FROM_VARIANT_AND_RETURN(0)
        EXTRACT_FROM_VARIANT_AND_RETURN(1)
        EXTRACT_FROM_VARIANT_AND_RETURN(2)
        EXTRACT_FROM_VARIANT_AND_RETURN(3)    
        EXTRACT_FROM_VARIANT_AND_RETURN(4)
        EXTRACT_FROM_VARIANT_AND_RETURN(5)
        EXTRACT_FROM_VARIANT_AND_RETURN(6)
        EXTRACT_FROM_VARIANT_AND_RETURN(7)    
        EXTRACT_FROM_VARIANT_AND_RETURN(8)
        EXTRACT_FROM_VARIANT_AND_RETURN(9)
    }
}

py::array get_image_data(bfiocpp::TsReaderCPP& tl, const Seq& rows, const Seq& cols, const Seq& layers, const Seq& channels, const Seq& tsteps) {
    auto tmp = tl.GetImageData(rows, cols, layers, channels, tsteps);
    auto ih = rows.Stop() - rows.Start() + 1;
    auto iw = cols.Stop() - cols.Start() + 1;
    auto id = layers.Stop() - layers.Start() + 1;;
    auto nc = channels.Stop() - channels.Start() + 1;
    auto nt = tsteps.Stop() - tsteps.Start() + 1;
 
    return as_pyarray_shared_5d(tmp, ih, iw, id, nc, nt) ;
}


py::array get_iterator_requested_tile_data(bfiocpp::TsReaderCPP& tl,  std::int64_t t_index,
                                                                        std::int64_t c_index,
                                                                        std::int64_t z_index,
                                                                        std::int64_t y_min, std::int64_t y_max,
                                                                        std::int64_t x_min, std::int64_t x_max) {

    auto rows = Seq(y_min, y_max, 1);
    auto cols = Seq(x_min, x_max, 1);
    auto layers = Seq(z_index, z_index, 1);
    auto channels = Seq(c_index, c_index, 1);
    auto tsteps = Seq(t_index, t_index, 1);

    auto tmp = tl.GetImageData(rows, cols, layers, channels, tsteps);
    auto ih = rows.Stop() - rows.Start() + 1;
    auto iw = cols.Stop() - cols.Start() + 1;
    auto id = layers.Stop() - layers.Start() + 1;;
    auto nc = channels.Stop() - channels.Start() + 1;
    auto nt = tsteps.Stop() - tsteps.Start() + 1;
 
    return as_pyarray_shared_5d(tmp, ih, iw, id, nc, nt) ;
}

PYBIND11_MODULE(libbfiocpp, m) {
    py::class_<Seq, std::shared_ptr<Seq>>(m, "Seq")  
        .def(py::init<const size_t, const size_t, const size_t>());
    
    py::class_<bfiocpp::TsReaderCPP, std::shared_ptr<bfiocpp::TsReaderCPP>>(m, "TsReaderCPP") 
    .def(py::init<const std::string &, bfiocpp::FileType, const std::string &>()) 
    .def("get_image_height", &bfiocpp::TsReaderCPP::GetImageHeight) 
    .def("get_image_width", &bfiocpp::TsReaderCPP::GetImageWidth) 
    .def("get_image_depth", &bfiocpp::TsReaderCPP::GetImageDepth) 
    .def("get_tile_height", &bfiocpp::TsReaderCPP::GetTileHeight) 
    .def("get_tile_width", &bfiocpp::TsReaderCPP::GetTileWidth) 
    .def("get_tile_depth", &bfiocpp::TsReaderCPP::GetTileDepth) 
    .def("get_channel_count", &bfiocpp::TsReaderCPP::GetChannelCount) 
    .def("get_tstep_count", &bfiocpp::TsReaderCPP::GetTstepCount) 
    .def("get_datatype", &bfiocpp::TsReaderCPP::GetDataType)
    .def("get_tile_coordinate",
        [](bfiocpp::TsReaderCPP& tl, std::int64_t y_start, std::int64_t x_start, std::int64_t row_stride, std::int64_t col_stride) { 
            auto row_index = static_cast<std::int64_t>(y_start/row_stride);
            auto col_index = static_cast<std::int64_t>(x_start/col_stride);
            return std::make_tuple(row_index, col_index);
        }
    )
    .def("get_image_data",  
        [](bfiocpp::TsReaderCPP& tl, const Seq& rows, const Seq& cols, const Seq& layers, const Seq& channels, const Seq& tsteps) { 
            return get_image_data(tl, rows, cols, layers, channels, tsteps);
        }, py::return_value_policy::reference) 
    .def("send_iterator_read_requests",
    [](bfiocpp::TsReaderCPP& tl, std::int64_t const tile_height, std::int64_t const tile_width, std::int64_t const row_stride, std::int64_t const col_stride) {
        tl.SetIterReadRequests(tile_height, tile_width, row_stride, col_stride);
    }) 
    .def("get_iterator_requested_tile_data", 
    [](bfiocpp::TsReaderCPP& tl,  std::int64_t t_index,
                                    std::int64_t c_index,
                                    std::int64_t z_index,
                                    std::int64_t y_min, std::int64_t y_max,
                                    std::int64_t x_min, std::int64_t x_max) {
        return get_iterator_requested_tile_data(tl, t_index, c_index, z_index, y_min, y_max, x_min, x_max);
    }, py::return_value_policy::reference)
    .def("__iter__", [](bfiocpp::TsReaderCPP& tl){ 
        return py::make_iterator(tl.iter_request_list.begin(), tl.iter_request_list.end());
        }, py::keep_alive<0, 1>()); 


    py::enum_<bfiocpp::FileType>(m, "FileType")
        .value("OmeTiff", bfiocpp::FileType::OmeTiff)
        .value("OmeZarr", bfiocpp::FileType::OmeZarr)
        .export_values();
    
    m.def("get_ome_xml", &bfiocpp::GetOmeXml);

    
    // Writer class
    py::class_<bfiocpp::TsWriterCPP, std::shared_ptr<bfiocpp::TsWriterCPP>>(m, "TsWriterCPP") 
    .def(py::init<const std::string&>())
    .def("write", &bfiocpp::TsWriterCPP::write_image);
}
