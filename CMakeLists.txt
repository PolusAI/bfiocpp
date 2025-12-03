cmake_minimum_required(VERSION 3.24)

project(bfio_cpp)
#==== Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
endif()

include(GNUInstallDirs)
# this is a workaround for GitHub Action for wheelbuiling
if(DEFINED ENV{BFIOCPP_DEP_DIR})
	set(CMAKE_PREFIX_PATH $ENV{BFIOCPP_DEP_DIR})
	link_directories($ENV{BFIOCPP_DEP_DIR}/${CMAKE_INSTALL_LIBDIR})
endif()
message(STATUS "CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
  
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(SOURCE 
          src/cpp/ts_driver/tiled_tiff/omexml.cc
          src/cpp/ts_driver/tiled_tiff/tiled_tiff_key_value_store.cc
          src/cpp/ts_driver/ometiff/metadata.cc
          src/cpp/ts_driver/ometiff/driver.cc
          src/cpp/interface/interface.cpp
          src/cpp/reader/tsreader.cpp
          src/cpp/utilities/utilities.cpp
          src/cpp/writer/tswriter.cpp
)

include(FetchContent)

FetchContent_Declare(
  tensorstore
  URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.36.tar.gz"
  URL_HASH SHA256=5857582b1b68e5a3f470d2ee8e9a7fa1ad6e6c7e0867c7c69e5a523ae00bc002
)

# Additional FetchContent_Declare calls as needed...
set(TENSORSTORE_USE_SYSTEM_GRPC ON CACHE BOOL "")
FetchContent_MakeAvailable(tensorstore)


include_directories(lib/pugixml)

find_package(Threads QUIET)
if (Threads_FOUND)
  if (CMAKE_USE_PTHREADS_INIT)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /pthread")
    endif()
  endif (CMAKE_USE_PTHREADS_INIT) 
    list(APPEND Build_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
else ()
	message(STATUS "Unable to find threads. bfio_cpp must have a threading library i.e. pthreads.")
endif ()

#==== Pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(libbfiocpp 
  ${SOURCE}
)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a define (VERSION_INFO) here.
target_compile_definitions(libbfiocpp PRIVATE VERSION_INFO=${VERSION_INFO})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
    target_link_libraries(libbfiocpp PRIVATE stdc++fs)
endif()

target_link_libraries(libbfiocpp PRIVATE 
                      tensorstore::tensorstore 
                      tensorstore::all_drivers)
target_link_libraries(libbfiocpp PRIVATE ${Build_LIBRARIES})  