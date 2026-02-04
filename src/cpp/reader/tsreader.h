#pragma once

#include <string>
#include <memory>
#include <vector>
#include <variant>
#include <iostream>
#include <tuple>
#include <optional>
#include <unordered_map>
#include "tensorstore/tensorstore.h"
#include "../utilities/sequence.h"
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


enum class FileType {OmeTiff, OmeZarrV2, OmeZarrV3};

class TsReaderCPP{
public:
    TsReaderCPP(const std::string& fname, FileType ft, const std::string& axes_list );
    std::int64_t GetImageHeight() const ;
    std::int64_t GetImageWidth () const ;
    std::int64_t GetImageDepth () const ;
    std::int64_t GetTileHeight () const ;
    std::int64_t GetTileWidth () const ;
    std::int64_t GetTileDepth () const ;
    std::int64_t GetChannelCount () const;
    std::int64_t GetTstepCount () const;
    std::string GetDataType() const;
    std::shared_ptr<image_data> GetImageData(const Seq& rows, const Seq& cols, const Seq& layers, const Seq& channels, const Seq& tsteps);
    void SetIterReadRequests(std::int64_t const tile_width, std::int64_t const tile_height, std::int64_t const row_stride, std::int64_t const col_stride);
    //tuple of (T,C,Z,Y_min, Y_max, X_min, X_max)
    std::vector<iter_indicies> iter_request_list;

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
    FileType _file_type;

    std::optional<int>_z_index, _c_index, _t_index;

    tensorstore::TensorStore<void, -1, tensorstore::ReadWriteMode::dynamic> source;


    template <typename T>
    std::shared_ptr<std::vector<T>> GetImageDataTemplated(const Seq& rows, const Seq& cols, const Seq& layers, const Seq& channels, const Seq& tsteps);                 
};
}

