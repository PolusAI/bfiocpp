#include <iomanip>
#include <ctime>
#include <chrono>
#include "utilities.h"
#include <cassert>
#include <tiffio.h>
#include <thread>

namespace bfiocpp {
tensorstore::Spec GetOmeTiffSpecToRead(const std::string& filename){
    return tensorstore::Spec::FromJson({{"driver", "ometiff"},

                            {"kvstore", {{"driver", "tiled_tiff"},
                                         {"path", filename}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", 8}}},
                              {"file_io_concurrency", {{"limit", 8}}},
                            }},
                            }).value();
}

tensorstore::Spec GetZarrSpecToRead(const std::string& filename){
    return tensorstore::Spec::FromJson({{"driver", "zarr"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", filename}}
                            }
                            }).value();
}


uint16_t GetDataTypeCode (std::string_view type_name){

  if (type_name == std::string_view{"uint8"}) {return 1;}
  else if (type_name == std::string_view{"uint16"}) {return 2;}
  else if (type_name == std::string_view{"uint32"}) {return 4;}
  else if (type_name == std::string_view{"uint64"}) {return 8;}
  else if (type_name == std::string_view{"int8"}) {return 16;}
  else if (type_name == std::string_view{"int16"}) {return 32;}
  else if (type_name == std::string_view{"int32"}) {return 64;}
  else if (type_name == std::string_view{"int64"}) {return 128;}
  else if (type_name == std::string_view{"float32"}) {return 256;}
  else if (type_name == std::string_view{"float64"}) {return 512;}
  else {return 2;}
}

std::string GetUTCString() {
    // Get the current UTC time
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);

    // Convert UTC time to a string
    const int bufferSize = 16; // Sufficient size for most date/time formats
    char buffer[bufferSize];
    std::tm timeInfo;

#if defined(_WIN32)
    // Use gmtime_s instead of gmtime to get the UTC time on Windows
    gmtime_s(&timeInfo, &time);
#else
    // On other platforms, use the standard gmtime function
    gmtime_r(&time, &timeInfo);
#endif
    // Format the time string (You can modify the format as per your requirements)
    std::strftime(buffer, bufferSize, "%Y%m%d%H%M%S", &timeInfo);

    return std::string(buffer);
}

std::tuple<std::optional<int>, std::optional<int>, std::optional<int>>ParseMultiscaleMetadata(const std::string& axes_list, int len){
    
    std::optional<int> t_index{std::nullopt}, c_index{std::nullopt}, z_index{std::nullopt};

    assert(axes_list.length() <= 5);

    if (axes_list.length() == len){
        // no speculation
        for (int i=0; i<axes_list.length(); ++i){
            if(axes_list[i] == char{'T'}) t_index.emplace(i);
            if(axes_list[i] == char{'C'}) c_index.emplace(i);
            if(axes_list[i] == char{'Z'}) z_index.emplace(i);
        }
    } else // speculate
    {
        if (len == 3) {
            z_index.emplace(0);
        } else if (len == 4) {
            z_index.emplace(1);
            c_index.emplace(0);
        }
    }
    return {t_index, c_index, z_index};
}


std::string GetOmeXml(const std::string& file_path){
    TIFF *tiff_file = TIFFOpen(file_path.c_str(), "r");
    std::string OmeXmlInfo{""};
    if (tiff_file != nullptr) {
        char* infobuf;        
        TIFFGetField(tiff_file, TIFFTAG_IMAGEDESCRIPTION , &infobuf);
        if (strlen(infobuf)>0){
            OmeXmlInfo = std::string(infobuf);
        }
        TIFFClose(tiff_file);
    }
    return OmeXmlInfo;
}

tensorstore::Spec GetZarrSpecToWrite(   const std::string& filename, 
                                        const std::vector<std::int64_t>& image_shape, 
                                        const std::vector<std::int64_t>& chunk_shape,
                                        const std::string& dtype){
    return tensorstore::Spec::FromJson({{"driver", "zarr"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", filename}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                              {"file_io_concurrency", {{"limit", std::thread::hardware_concurrency()}}},
                            }},
                            {"metadata", {
                                          {"zarr_format", 2},
                                          {"shape", image_shape},
                                          {"chunks", chunk_shape},
                                          {"dtype", dtype},
                                          },
                            }}).value();
                                        }

// Function to get the TensorStore DataType based on a string identifier
tensorstore::DataType GetTensorStoreDataType(const std::string& type_str) {
    if (type_str == "uint8") {
        return tensorstore::dtype_v<std::uint8_t>;
    } else if (type_str == "uint16") {
        return tensorstore::dtype_v<std::uint16_t>;
    } else if (type_str == "uint32") {
        return tensorstore::dtype_v<std::uint32_t>;
    } else if (type_str == "uint64") {
        return tensorstore::dtype_v<std::uint64_t>;
    } else if (type_str == "int8") {
        return tensorstore::dtype_v<std::int8_t>;
    } else if (type_str == "int16") {
        return tensorstore::dtype_v<std::int16_t>;
    } else if (type_str == "int32") {
        return tensorstore::dtype_v<std::int32_t>;
    } else if (type_str == "int64") {
        return tensorstore::dtype_v<std::int64_t>;
    } else if (type_str == "float") {
        return tensorstore::dtype_v<float>;
    } else if (type_str == "double" || type_str == "float64") { // handle float64 from numpy
        return tensorstore::dtype_v<double>;
    } else {
        throw std::invalid_argument("Unknown data type string: " + type_str);
    }
}

} // ns bfiocpp