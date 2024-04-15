
#pragma once
#include <string>
#include <memory>
#include <vector>
#include<cmath>

#include "tensorstore/tensorstore.h"
#include "tensorstore/spec.h"

namespace bfiocpp {

tensorstore::Spec GetOmeTiffSpecToRead(const std::string& filename);



uint16_t GetDataTypeCode (std::string_view type_name);
std::string GetUTCString();
} // ns bfiocpp