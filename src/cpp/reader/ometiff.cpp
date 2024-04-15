#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"

#include "ometiff.h"
#include "utilities.h"
#include "type_info.h"
#include <tiffio.h>


using ::tensorstore::internal_zarr::ChooseBaseDType;

namespace bfiocpp{

OmeTiffReader::OmeTiffReader(const std::string& fname){
    _filename = fname;
    TENSORSTORE_CHECK_OK_AND_ASSIGN(source, tensorstore::Open(
                GetOmeTiffSpecToRead(_filename),
                tensorstore::OpenMode::open,
                tensorstore::ReadWriteMode::read).result());
    
    auto image_shape = source.domain().shape();
    
    _image_height = image_shape[3];
    _image_width = image_shape[4];
    _image_depth = image_shape[2];    
    _num_channels = image_shape[1];
    _num_tsteps = image_shape[0];

    const auto read_chunk_shape = source.chunk_layout().value().read_chunk_shape();
    _tile_height = static_cast<std::int64_t>(read_chunk_shape[3]);
    _tile_width = static_cast<std::int64_t>(read_chunk_shape[4]);  
    
    _data_type = ChooseBaseDType(source.dtype()).value().encoded_dtype;
    _data_type_code = GetDataTypeCode(_data_type);
}

std::int64_t OmeTiffReader::GetImageHeight() const {return _image_height;} 
std::int64_t OmeTiffReader::GetImageDepth() const {return _image_depth;} 
std::int64_t OmeTiffReader::GetImageWidth() const {return _image_width;} 
std::int64_t OmeTiffReader::GetTileHeight() const {return _tile_height;} 
std::int64_t OmeTiffReader::GetTileDepth() const {return _tile_depth;} 
std::int64_t OmeTiffReader::GetTileWidth() const {return _tile_width;} 
std::int64_t OmeTiffReader::GetChannelCount() const {return _num_channels;} 
std::int64_t OmeTiffReader::GetTstepCount() const {return _num_tsteps;} 
std::string OmeTiffReader::GetDataType() const {return _data_type;} 

template <typename T>
std::shared_ptr<std::vector<T>> OmeTiffReader::GetImageDataTemplated(const Seq& rows, const Seq& cols, const Seq& layers, const Seq& channels, const Seq& tsteps){

    const auto data_height = rows.Stop() - rows.Start() + 1;
    const auto data_width = cols.Stop() - cols.Start() + 1;
    const auto data_depth = layers.Stop() - layers.Start() + 1;
    const auto data_num_channels = channels.Stop() - channels.Start() + 1;
    const auto data_tsteps = tsteps.Stop() - tsteps.Start() + 1;

    auto read_buffer = std::make_shared<std::vector<T>>(data_height*data_width*data_depth*data_num_channels*data_tsteps); 
    auto array = tensorstore::Array(read_buffer->data(), {data_tsteps, data_num_channels, data_depth, data_height, data_width}, tensorstore::c_order);
    tensorstore::Read(source | 
        tensorstore::Dims(0).ClosedInterval(tsteps.Start(), tsteps.Stop()) |
        tensorstore::Dims(1).ClosedInterval(channels.Start(), channels.Stop()) |
        tensorstore::Dims(2).ClosedInterval(layers.Start(), layers.Stop()) |
        tensorstore::Dims(3).ClosedInterval(rows.Start(), rows.Stop()) |
        tensorstore::Dims(4).ClosedInterval(cols.Start(), cols.Stop()),
        tensorstore::UnownedToShared(array)).value();

    return read_buffer;
}

std::shared_ptr<image_data> OmeTiffReader::GetImageData(const Seq& rows, const Seq& cols, const Seq& layers = Seq(0,0), const Seq& channels = Seq(0,0), const Seq& tsteps = Seq(0,0)) {
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

std::string OmeTiffReader::GetOmeXml() const{
    TIFF *tiff_file = TIFFOpen(_filename.c_str(), "r");
    std::string OmeXmlInfo{""};
    if (tiff_file != nullptr) {
        char* infobuf;        
        TIFFGetField(tiff_file, TIFFTAG_IMAGEDESCRIPTION , &infobuf);
        if (strlen(infobuf)>0){
            OmeXmlInfo = std::string(infobuf);
        }
    }
    TIFFClose(tiff_file);
    return OmeXmlInfo;
}

void OmeTiffReader::SetIterReadRequests(std::int64_t const tile_width, std::int64_t const tile_height, std::int64_t const row_stride, std::int64_t const col_stride){
    iter_request_list.clear();
    for(std::int64_t t=0; t<_num_tsteps;++t){
        for(std::int64_t c=0; c<_num_channels;++c){
            for(std::int64_t z=0; z<_image_depth;++z){
                for(std::int64_t y=0; y<=_image_height;y+=row_stride)
                {
                    auto y_min = y;
                    auto y_max = y_min + row_stride - 1;
                    y_max = y_max < _image_height ? y_max : _image_height-1;

                    for(std::int64_t x=0; x<=_image_width;x+=col_stride){
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