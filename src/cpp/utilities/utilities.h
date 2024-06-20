
#pragma once
#include <string>
#include <memory>
#include <vector>
#include<cmath>
#include <tuple>
#include <optional>
#include "tensorstore/tensorstore.h"
#include "tensorstore/spec.h"

namespace bfiocpp {

tensorstore::Spec GetOmeTiffSpecToRead(const std::string& filename);
tensorstore::Spec GetZarrSpecToRead(const std::string& filename);

uint16_t GetDataTypeCode (std::string_view type_name);
std::string GetUTCString();
std::string GetOmeXml(const std::string& file_path);
std::tuple<std::optional<int>, std::optional<int>, std::optional<int>>ParseMultiscaleMetadata(const std::string& axes_list, int len);
tensorstore::DataType GetTensorStoreDataType(const std::string& type_str);
uint16_t GetDataTypeCode (std::string_view type_name);
tensorstore::Spec GetZarrSpecToWrite(const std::string& filename, 
                                    const std::vector<std::int64_t>& image_shape, 
                                    const std::vector<std::int64_t>& chunk_shape,
                                    const std::string& dtype);
} // ns bfiocpp