
#include "metadata.h"

#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_spec.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/transform_broadcastable_array.h"
#include "tensorstore/internal/async_write_array.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/future.h"

#include <iostream>
#include <numeric>
#include <tuple>
#include <string>

namespace tensorstore {
namespace internal_ometiff {

namespace {

namespace jb = tensorstore::internal_json_binding;

constexpr const char kMetadataKey[] = "IMAGE_DESCRIPTION";

using internal_kvs_backed_chunk_driver::KvsDriverSpec;

class OmeTiffDriverSpec
    : public internal::RegisteredDriverSpec<OmeTiffDriverSpec,
                                            /*Parent=*/KvsDriverSpec> {  

 public:
  constexpr static char id[] = "ometiff";

  using Base = internal::RegisteredDriverSpec<OmeTiffDriverSpec,
                                              /*Parent=*/KvsDriverSpec>;

  OmeTiffMetadataConstraints metadata_constraints;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<KvsDriverSpec>(x), x.metadata_constraints);
  };

  static inline const auto default_json_binder = jb::Sequence(
      jb::Validate([](const auto& options, auto* obj) {
            if (obj->schema.dtype().valid()) {
              return ValidateDataType(obj->schema.dtype());
            }
            return absl::OkStatus();
          },
          internal_kvs_backed_chunk_driver::SpecJsonBinder),
      jb::Member(
          "metadata",
          jb::Validate( 
              [](const auto& options, auto* obj) {
                TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(
                    obj->metadata_constraints.dtype.value_or(DataType())));
                TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(
                    RankConstraint{obj->metadata_constraints.rank}));
                return absl::OkStatus();
              },
              jb::Projection<&OmeTiffDriverSpec::metadata_constraints>(
                  jb::DefaultInitializedValue()))));
  
  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.minimal_spec) {
      metadata_constraints = OmeTiffMetadataConstraints{};
    }
    return Base::ApplyOptions(std::move(options));
  }
  Result<IndexDomain<>> GetDomain() const override {
    return GetEffectiveDomain(metadata_constraints, schema);
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    return GetEffectiveChunkLayout(metadata_constraints, schema);
  }
  

  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override;
};

// we need OMETiff Metadata 
Result<std::shared_ptr<const OmeTiffMetadata>> ParseEncodedMetadata(
    std::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  // create ifd lookup table
  std::map<std::tuple<size_t, size_t, size_t>, size_t> ifd_lookup_table;

  for(auto &el : raw_data["omeXml"]["tiffData"].items()){
    ifd_lookup_table.emplace(el.value().get<std::tuple<size_t,size_t,size_t>>(), 
                              std::stoi(el.key()));
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               OmeTiffMetadata::FromJson(std::move(raw_data)));
  

  metadata.ifd_lookup_table = ifd_lookup_table;
  return std::make_shared<OmeTiffMetadata>(std::move(metadata));
}

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;


 public:
  using Base::Base;

  // Metadata is stored as IMAGE_DESCRIPTION tag inside tiff.
  std::string GetMetadataStorageKey(std::string_view entry_key) override {
    // metadata is in the same file
    return tensorstore::StrCat(entry_key, "__TAG__/", kMetadataKey);
//    return std::string(entry_key);
  }
  Result<MetadataPtr> DecodeMetadata(std::string_view entry_key,
                                     absl::Cord encoded_metadata) override {
    return ParseEncodedMetadata(encoded_metadata.Flatten());
  }

  Result<absl::Cord> EncodeMetadata(std::string_view entry_key,
                                    const void* metadata) override {
    return absl::Cord(
        ::nlohmann::json(*static_cast<const OmeTiffMetadata*>(metadata)).dump());
  }
};

class DataCache : public internal_kvs_backed_chunk_driver::DataCache {
  using Base = internal_kvs_backed_chunk_driver::DataCache;

 public:
  explicit DataCache(Initializer&& initializer, std::string key_prefix)
      : Base(std::move(initializer),
             GetChunkGridSpecification(
                 *static_cast<const OmeTiffMetadata*>(
                     initializer.metadata.get()))),
        key_prefix_(std::move(key_prefix)) {}

  const OmeTiffMetadata& metadata() const {
    return *static_cast<const OmeTiffMetadata*>(initial_metadata().get());
  }

  absl::Status ValidateMetadataCompatibility(
      const void* existing_metadata_ptr,
      const void* new_metadata_ptr) override {
    const auto& existing_metadata =
        *static_cast<const OmeTiffMetadata*>(existing_metadata_ptr);
    const auto& new_metadata =
        *static_cast<const OmeTiffMetadata*>(new_metadata_ptr);
    auto existing_key = existing_metadata.GetCompatibilityKey();
    auto new_key = new_metadata.GetCompatibilityKey();
    if (existing_key == new_key) return absl::OkStatus();
    return absl::FailedPreconditionError(
        StrCat("Updated OmeTiff metadata ", new_key,
               " is incompatible with existing metadata ", existing_key));
  }

  void GetChunkGridBounds(const void* metadata_ptr, MutableBoxView<> bounds,
                          DimensionSet& implicit_lower_bounds,
                          DimensionSet& implicit_upper_bounds) override {
    const auto& metadata = *static_cast<const OmeTiffMetadata*>(metadata_ptr);
    assert(bounds.rank() == static_cast<DimensionIndex>(metadata.shape.size()));
    std::fill(bounds.origin().begin(), bounds.origin().end(), Index(0));
    std::copy(metadata.shape.begin(), metadata.shape.end(),
              bounds.shape().begin());
    implicit_lower_bounds = false;
    implicit_upper_bounds = true;
  }

  Result<std::shared_ptr<const void>> GetResizedMetadata(
      const void* existing_metadata, span<const Index> new_inclusive_min,
      span<const Index> new_exclusive_max) override {
    auto new_metadata = std::make_shared<OmeTiffMetadata>(
        *static_cast<const OmeTiffMetadata*>(existing_metadata));
    const DimensionIndex rank = new_metadata->shape.size();
    assert(rank == new_inclusive_min.size());
    assert(rank == new_exclusive_max.size());
    for (DimensionIndex i = 0; i < rank; ++i) {
      assert(ExplicitIndexOr(new_inclusive_min[i], 0) == 0);
      const Index new_size = new_exclusive_max[i];
      if (new_size == kImplicit) continue;
      new_metadata->shape[i] = new_size;
    }
    return new_metadata;
  }

  static internal::ChunkGridSpecification GetChunkGridSpecification(
      const OmeTiffMetadata& metadata) {
    const DimensionIndex rank = metadata.rank;

    // Create fill value (zero-initialized)
    auto fill_value = AllocateArray(span<const Index, 0>{}, c_order, value_init,
                                    metadata.dtype);

    // Valid data bounds - unbounded since all dimensions are resizable
    Box<> valid_data_bounds(rank);

    // Broadcast fill value to the valid bounds
    auto chunk_fill_value = BroadcastArray(fill_value, valid_data_bounds).value();

    // Cell chunk shape from metadata
    std::vector<Index> cell_chunk_shape(metadata.chunk_layout.shape().begin(),
                                        metadata.chunk_layout.shape().end());

    // Chunked to cell dimensions mapping (identity for OmeTiff)
    std::vector<DimensionIndex> chunked_to_cell_dimensions(rank);
    std::iota(chunked_to_cell_dimensions.begin(),
              chunked_to_cell_dimensions.end(), static_cast<DimensionIndex>(0));

    // Create layout order buffer for C order (identity permutation)
    DimensionIndex layout_order_buffer[kMaxRank];
    for (DimensionIndex i = 0; i < rank; ++i) {
      layout_order_buffer[i] = i;
    }

    // Build component list with AsyncWriteArray::Spec
    internal::ChunkGridSpecification::ComponentList components;
    components.emplace_back(
        internal::AsyncWriteArray::Spec{std::move(chunk_fill_value),
                                        std::move(valid_data_bounds),
                                        ContiguousLayoutPermutation<>(
                                            span(layout_order_buffer, rank))},
        std::move(cell_chunk_shape), chunked_to_cell_dimensions);

    return internal::ChunkGridSpecification{std::move(components)};
  }

  Result<absl::InlinedVector<SharedArray<const void>, 1>> DecodeChunk(
      span<const Index> chunk_indices,
      absl::Cord data) override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto array,
        internal_ometiff::DecodeChunk(metadata(), std::move(data)));
    return absl::InlinedVector<SharedArray<const void>, 1>{
        SharedArray<const void>(std::move(array))};
  }


  Result<absl::Cord> EncodeChunk(
      span<const Index> chunk_indices,
      span<const SharedArray<const void>> component_arrays) override {
    return absl::Cord();
  }

   std::string GetChunkStorageKey(span<const Index> cell_indices) override {
    // OMETiff is always 5D. So need to add some check here
    const auto& md = metadata();

    size_t ifd = md.GetIfdIndex(cell_indices[2],cell_indices[1],cell_indices[0]);
    std::string key =
         StrCat(key_prefix_, "__TAG__/" );
    auto& chunk_shape = md.chunk_shape;
//    StrAppend(&key, cell_indices.empty() ? 0 : cell_indices[0]);

    StrAppend(&key, "_", cell_indices[3]*chunk_shape[3]);
    StrAppend(&key, "_", cell_indices[4]*chunk_shape[4]);
    StrAppend(&key, "_", ifd);
    return key;
  }

  Result<IndexTransform<>> GetExternalToInternalTransform(
      const void* metadata_ptr, std::size_t component_index) override {
    assert(component_index == 0);
    const auto& metadata = *static_cast<const OmeTiffMetadata*>(metadata_ptr);

    const DimensionIndex rank = metadata.shape.size();
    auto builder = tensorstore::IndexTransformBuilder<>(rank, rank)
                       .input_shape(metadata.shape);
                       //.input_labels(axes);
    builder.implicit_upper_bounds(true);
    for (DimensionIndex i = 0; i < rank; ++i) {
      builder.output_single_input_dimension(i, i);
    }
    return builder.Finalize();
  }
  absl::Status GetBoundSpecData(KvsDriverSpec& spec_base,
                                const void* metadata_ptr,
                                std::size_t component_index) override {
    assert(component_index == 0);
    auto& spec = static_cast<OmeTiffDriverSpec&>(spec_base);
    const auto& metadata = *static_cast<const OmeTiffMetadata*>(metadata_ptr);
    auto& constraints = spec.metadata_constraints;
    constraints.shape = metadata.shape;
    constraints.dtype = metadata.dtype;
    constraints.dim_order = metadata.dim_order;
    constraints.extra_attributes = metadata.extra_attributes;
    constraints.chunk_shape =
        std::vector<Index>(metadata.chunk_layout.shape().begin(),
                           metadata.chunk_layout.shape().end());
    return absl::OkStatus();
  }

  Result<ChunkLayout> GetChunkLayoutFromMetadata(const void* metadata_ptr,
                                                  size_t component_index) override {
    const auto& md = *static_cast<const OmeTiffMetadata*>(metadata_ptr);
    ChunkLayout chunk_layout;
    TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
        md.rank, md.chunk_shape, chunk_layout));
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Finalize());
    return chunk_layout;
  }

  std::string GetBaseKvstorePath() override { return key_prefix_; }

 private:
  std::string key_prefix_;
};

// Forward declaration for the mixin
class OmeTiffDriver;

using OmeTiffDriverBase = internal_kvs_backed_chunk_driver::RegisteredKvsDriver<
    OmeTiffDriver, OmeTiffDriverSpec, DataCache,
    internal::ChunkCacheReadWriteDriverMixin<
        OmeTiffDriver, internal_kvs_backed_chunk_driver::KvsChunkedDriverBase>>;

class OmeTiffDriver : public OmeTiffDriverBase {
  using Base = OmeTiffDriverBase;

 public:
  using Base::Base;

  class OpenState;

};

class OmeTiffDriver::OpenState : public OmeTiffDriver::OpenStateBase {
 public:
  using OmeTiffDriver::OpenStateBase::OpenStateBase;

  std::string GetPrefixForDeleteExisting() override {
    return spec().store.path;
  }
  std::string GetMetadataCacheEntryKey() override { 
    return spec().store.path; 
  }
  
  std::unique_ptr<internal_kvs_backed_chunk_driver::MetadataCache>
  GetMetadataCache(MetadataCache::Initializer initializer) override {
    return std::make_unique<MetadataCache>(std::move(initializer));
  }

  std::string GetDataCacheKey(const void* metadata) override {
    std::string result;
    internal::EncodeCacheKey(
        &result, spec().store.path,
        static_cast<const OmeTiffMetadata*>(metadata)->GetCompatibilityKey());
    return result;
  }




  Result<std::shared_ptr<const void>> Create(
      const void* existing_metadata, CreateOptions options) override {
    if (existing_metadata) {
      return absl::AlreadyExistsError("");
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto metadata,
        internal_ometiff::GetNewMetadata(spec().metadata_constraints, spec().schema),
        tensorstore::MaybeAnnotateStatus(
            _, "Cannot create using specified \"metadata\" and schema"));
    return metadata;
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCacheBase> GetDataCache(
      internal_kvs_backed_chunk_driver::DataCacheInitializer&& initializer) override {
    return std::make_unique<DataCache>(std::move(initializer),
                                       spec().store.path);
  }

  Result<std::size_t> GetComponentIndex(const void* metadata_ptr,
                                        OpenMode open_mode) override {
    const auto& metadata = *static_cast<const OmeTiffMetadata*>(metadata_ptr);
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateMetadata(metadata, spec().metadata_constraints));
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateMetadataSchema(metadata, spec().schema));
    return 0;
  }

};

Future<internal::Driver::Handle> OmeTiffDriverSpec::Open(
    internal::DriverOpenRequest request) const {
  return OmeTiffDriver::Open(this, std::move(request));
}

}  // namespace
}  // namespace internal_ometiff
}  // namespace tensorstore


TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_ometiff::OmeTiffDriver)
// Use default garbage collection implementation provided by
// kvs_backed_chunk_driver (just handles the kvstore)
TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_ometiff::OmeTiffDriver,
    tensorstore::internal_ometiff::OmeTiffDriver::GarbageCollectionBase)

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_ometiff::OmeTiffDriverSpec>
    registration;
}  // namespace