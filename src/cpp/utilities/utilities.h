
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

enum class FileType {OmeTiff, OmeZarrV2, OmeZarrV3};

tensorstore::Spec GetOmeTiffSpecToRead(const std::string& filename);
tensorstore::Spec GetZarrSpecToRead(const std::string& filename, FileType ft);

uint16_t GetDataTypeCode (std::string_view type_name);
std::string GetEncodedType(uint16_t data_type_code);
std::string GetUTCString();
std::string GetOmeXml(const std::string& file_path);
std::tuple<std::optional<int>, std::optional<int>, std::optional<int>>ParseMultiscaleMetadata(const std::string& axes_list, int len);
tensorstore::Spec GetZarrSpecToWrite(const std::string& filename,
                                    const std::vector<std::int64_t>& image_shape,
                                    const std::vector<std::int64_t>& chunk_shape,
                                    const std::string& dtype,
                                    FileType ft);
std::string GetZarrV3DataType(uint16_t data_type_code);
} // ns bfiocpp