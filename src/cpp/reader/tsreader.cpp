#include <cassert>
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"

#include "tsreader.h"
#include "../utilities/utilities.h"
#include "type_info.h"


using ::tensorstore::internal_zarr::ChooseBaseDType;

namespace bfiocpp{

TsReaderCPP::TsReaderCPP(const std::string& fname, FileType ft, const std::string& axes_list=""): _filename(fname), _file_type (ft) {

    auto read_spec = [fname, ft](){
        if (ft == FileType::OmeTiff){
            return GetOmeTiffSpecToRead(fname);
        } else {
            return GetZarrSpecToRead(fname, ft);
        }
    }();

    TENSORSTORE_CHECK_OK_AND_ASSIGN(source, tensorstore::Open(
                read_spec,
                tensorstore::OpenMode::open,
                tensorstore::ReadWriteMode::read).result());
    
    auto image_shape = source.domain().shape();
    const auto read_chunk_shape = source.chunk_layout().value().read_chunk_shape();
    if (_file_type == FileType::OmeTiff){
        assert(image_shape.size() == 5);
        _image_height = image_shape[3];
        _image_width = image_shape[4];
        _image_depth = image_shape[2];    
        _num_channels = image_shape[1];
        _num_tsteps = image_shape[0];

        assert(read_chunk_shape.size() == 5);

        _tile_height = static_cast<std::int64_t>(read_chunk_shape[3]);
        _tile_width = static_cast<std::int64_t>(read_chunk_shape[4]);  

        _t_index.emplace(0);
        _c_index.emplace(1);
        _z_index.emplace(2);
    } else {
        if (image_shape.size() == 5){
            _image_height = image_shape[3];
            _image_width = image_shape[4];
            _image_depth = image_shape[2];    
            _num_channels = image_shape[1];
            _num_tsteps = image_shape[0];
            _t_index.emplace(0);
            _c_index.emplace(1);
            _z_index.emplace(2);
        } else {
            assert(image_shape.size() >= 2);
            std::tie(_t_index, _c_index, _z_index) = ParseMultiscaleMetadata(axes_list, image_shape.size());
            _image_height = image_shape[image_shape.size()-2];
            _image_width = image_shape[image_shape.size()-1];
            if (_t_index.has_value()) {
                _num_tsteps = image_shape[_t_index.value()];
            } else {
                _num_tsteps = 1;
            }

            if (_c_index.has_value()) {
                _num_channels = image_shape[_c_index.value()];
            } else {
                _num_channels = 1;
            }

            if (_z_index.has_value()) {
                _image_depth = image_shape[_z_index.value()];
            } else {
                _image_depth = 1;
            }
        }
    }
    

    _data_type = source.dtype().name();
    _data_type_code = GetDataTypeCode(_data_type);
}

std::int64_t TsReaderCPP::GetImageHeight() const {return _image_height;} 
std::int64_t TsReaderCPP::GetImageDepth() const {return _image_depth;} 
std::int64_t TsReaderCPP::GetImageWidth() const {return _image_width;} 
std::int64_t TsReaderCPP::GetTileHeight() const {return _tile_height;} 
std::int64_t TsReaderCPP::GetTileDepth() const {return _tile_depth;} 
std::int64_t TsReaderCPP::GetTileWidth() const {return _tile_width;} 
std::int64_t TsReaderCPP::GetChannelCount() const {return _num_channels;} 
std::int64_t TsReaderCPP::GetTstepCount() const {return _num_tsteps;} 
std::string TsReaderCPP::GetDataType() const {return _data_type;} 

template <typename T>
std::shared_ptr<std::vector<T>> TsReaderCPP::GetImageDataTemplated(const Seq& rows, const Seq& cols, const Seq& layers, const Seq& channels, const Seq& tsteps){

    const auto data_height = rows.Stop() - rows.Start() + 1;
    const auto data_width = cols.Stop() - cols.Start() + 1;
    const auto data_depth = layers.Stop() - layers.Start() + 1;
    const auto data_num_channels = channels.Stop() - channels.Start() + 1;
    const auto data_tsteps = tsteps.Stop() - tsteps.Start() + 1;

    auto read_buffer = std::make_shared<std::vector<T>>(data_height*data_width*data_depth*data_num_channels*data_tsteps); 
    tensorstore::IndexTransform<> read_transform = tensorstore::IdentityTransform(source.domain());

    if (_file_type == FileType::OmeTiff) {
        read_transform = (std::move(read_transform) | tensorstore::Dims(0).ClosedInterval(tsteps.Start(), tsteps.Stop()) |
                                                        tensorstore::Dims(1).ClosedInterval(channels.Start(), channels.Stop()) |
                                                        tensorstore::Dims(2).ClosedInterval(layers.Start(), layers.Stop()) |
                                                        tensorstore::Dims(3).ClosedInterval(rows.Start(), rows.Stop()) |
                                                        tensorstore::Dims(4).ClosedInterval(cols.Start(), cols.Stop())).value(); 

        auto array = tensorstore::Array(read_buffer->data(), {data_tsteps, data_num_channels, data_depth, data_height, data_width}, tensorstore::c_order);
        tensorstore::Read(source | read_transform, tensorstore::UnownedToShared(array)).value();
    } else {
        std::vector<std::int64_t> array_shape;
        array_shape.reserve(5); 
        auto source_shape = source.domain().shape();
        int x_index = static_cast<int>(source_shape.size()) - 1; 
        int y_index = static_cast<int>(source_shape.size()) - 2;

        if (_t_index.has_value()){
            read_transform = (std::move(read_transform) | tensorstore::Dims(_t_index.value()).ClosedInterval(tsteps.Start(), tsteps.Stop())).value();
            array_shape.push_back(data_tsteps);
        }
        if (_c_index.has_value()){
            read_transform = (std::move(read_transform) | tensorstore::Dims(_c_index.value()).ClosedInterval(channels.Start(), channels.Stop())).value();
            array_shape.push_back(data_num_channels);
        }
        if (_z_index.has_value()){
            read_transform = (std::move(read_transform) | tensorstore::Dims(_z_index.value()).ClosedInterval(layers.Start(), layers.Stop())).value();
            array_shape.push_back(data_depth);
        }
        read_transform = (std::move(read_transform) | tensorstore::Dims(y_index).ClosedInterval(rows.Start(), rows.Stop()) |
                                                    tensorstore::Dims(x_index).ClosedInterval(cols.Start(), cols.Stop())).value(); 
        
        array_shape.push_back(data_height);
        array_shape.push_back(data_width);
        
        auto array = tensorstore::Array(read_buffer->data(), array_shape, tensorstore::c_order);
        tensorstore::Read(source | read_transform, tensorstore::UnownedToShared(array)).value();
    }

    return read_buffer;
}


std::shared_ptr<image_data> TsReaderCPP::GetImageData(const Seq& rows, const Seq& cols, const Seq& layers = Seq(0,0), const Seq& channels = Seq(0,0), const Seq& tsteps = Seq(0,0)) {
    switch (_data_type_code)
    {
    case (1):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<std::uint8_t>(rows, cols, layers, channels, tsteps))));
        break;
    case (2):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<std::uint16_t>(rows, cols, layers, channels, tsteps))));
        break;    
    case (4):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<std::uint32_t>(rows, cols, layers, channels, tsteps))));
        break;
    case (8):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<std::uint64_t>(rows, cols, layers, channels, tsteps))));
        break;
    case (16):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<std::int8_t>(rows, cols, layers, channels, tsteps))));
        break;
    case (32):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<std::int16_t>(rows, cols, layers, channels, tsteps))));
        break;    
    case (64):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<std::int32_t>(rows, cols, layers, channels, tsteps))));
        break;
    case (128):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<std::int64_t>(rows, cols, layers, channels, tsteps))));
        break;
    case (256):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<float>(rows, cols, layers, channels, tsteps))));
        break;
    case (512):
        return std::make_shared<image_data>(std::move(*(GetImageDataTemplated<double>(rows, cols, layers, channels, tsteps))));
        break;
    default:
        break;
    }
} 


void TsReaderCPP::SetIterReadRequests(std::int64_t const tile_width, std::int64_t const tile_height, std::int64_t const row_stride, std::int64_t const col_stride){
    iter_request_list.clear();
    for(std::int64_t t=0; t<_num_tsteps;++t){
        for(std::int64_t c=0; c<_num_channels;++c){
            for(std::int64_t z=0; z<_image_depth;++z){
                for(std::int64_t y=0; y<_image_height;y+=row_stride)
                {
                    auto y_min = y;
                    auto y_max = y_min + row_stride - 1;
                    y_max = y_max < _image_height ? y_max : _image_height-1;

                    for(std::int64_t x=0; x<_image_width;x+=col_stride){
                        auto x_min = x;
                        auto x_max = x_min + col_stride - 1;
                        x_max = x_max < _image_width ? x_max : _image_width-1;
                        iter_request_list.emplace_back(t,c,z,y_min,y_max,x_min,x_max);

                    }
                }
            }

        }
    }
}
}