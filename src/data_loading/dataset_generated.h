// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_DATASET_DLFS_H_
#define FLATBUFFERS_GENERATED_DATASET_DLFS_H_

#include "flatbuffers/flatbuffers.h"

namespace DLFS {

struct BoundingBox;

struct Category;

struct Annotation;

struct Example;

struct Dataset;

FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(4) BoundingBox FLATBUFFERS_FINAL_CLASS {
 private:
  float y1_;
  float x1_;
  float y2_;
  float x2_;

 public:
  BoundingBox() {
    memset(static_cast<void *>(this), 0, sizeof(BoundingBox));
  }
  BoundingBox(float _y1, float _x1, float _y2, float _x2)
      : y1_(flatbuffers::EndianScalar(_y1)),
        x1_(flatbuffers::EndianScalar(_x1)),
        y2_(flatbuffers::EndianScalar(_y2)),
        x2_(flatbuffers::EndianScalar(_x2)) {
  }
  float y1() const {
    return flatbuffers::EndianScalar(y1_);
  }
  void mutate_y1(float _y1) {
    flatbuffers::WriteScalar(&y1_, _y1);
  }
  float x1() const {
    return flatbuffers::EndianScalar(x1_);
  }
  void mutate_x1(float _x1) {
    flatbuffers::WriteScalar(&x1_, _x1);
  }
  float y2() const {
    return flatbuffers::EndianScalar(y2_);
  }
  void mutate_y2(float _y2) {
    flatbuffers::WriteScalar(&y2_, _y2);
  }
  float x2() const {
    return flatbuffers::EndianScalar(x2_);
  }
  void mutate_x2(float _x2) {
    flatbuffers::WriteScalar(&x2_, _x2);
  }
};
FLATBUFFERS_STRUCT_END(BoundingBox, 16);

struct Category FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_ID = 6
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  flatbuffers::String *mutable_name() {
    return GetPointer<flatbuffers::String *>(VT_NAME);
  }
  uint16_t id() const {
    return GetField<uint16_t>(VT_ID, 0);
  }
  bool mutate_id(uint16_t _id) {
    return SetField<uint16_t>(VT_ID, _id, 0);
  }
  bool KeyCompareLessThan(const Category *o) const {
    return id() < o->id();
  }
  int KeyCompareWithValue(uint16_t val) const {
    return static_cast<int>(id() > val) - static_cast<int>(id() < val);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyField<uint16_t>(verifier, VT_ID) &&
           verifier.EndTable();
  }
};

struct CategoryBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(Category::VT_NAME, name);
  }
  void add_id(uint16_t id) {
    fbb_.AddElement<uint16_t>(Category::VT_ID, id, 0);
  }
  explicit CategoryBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  CategoryBuilder &operator=(const CategoryBuilder &);
  flatbuffers::Offset<Category> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Category>(end);
    return o;
  }
};

inline flatbuffers::Offset<Category> CreateCategory(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    uint16_t id = 0) {
  CategoryBuilder builder_(_fbb);
  builder_.add_name(name);
  builder_.add_id(id);
  return builder_.Finish();
}

inline flatbuffers::Offset<Category> CreateCategoryDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    uint16_t id = 0) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  return DLFS::CreateCategory(
      _fbb,
      name__,
      id);
}

struct Annotation FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_BBOX = 4,
    VT_CAT_ID = 6,
    VT_ID = 8,
    VT_IMAGE_ID = 10
  };
  const DLFS::BoundingBox *bbox() const {
    return GetStruct<const DLFS::BoundingBox *>(VT_BBOX);
  }
  DLFS::BoundingBox *mutable_bbox() {
    return GetStruct<DLFS::BoundingBox *>(VT_BBOX);
  }
  uint16_t cat_id() const {
    return GetField<uint16_t>(VT_CAT_ID, 0);
  }
  bool mutate_cat_id(uint16_t _cat_id) {
    return SetField<uint16_t>(VT_CAT_ID, _cat_id, 0);
  }
  uint64_t id() const {
    return GetField<uint64_t>(VT_ID, 0);
  }
  bool mutate_id(uint64_t _id) {
    return SetField<uint64_t>(VT_ID, _id, 0);
  }
  bool KeyCompareLessThan(const Annotation *o) const {
    return id() < o->id();
  }
  int KeyCompareWithValue(uint64_t val) const {
    return static_cast<int>(id() > val) - static_cast<int>(id() < val);
  }
  uint64_t image_id() const {
    return GetField<uint64_t>(VT_IMAGE_ID, 0);
  }
  bool mutate_image_id(uint64_t _image_id) {
    return SetField<uint64_t>(VT_IMAGE_ID, _image_id, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<DLFS::BoundingBox>(verifier, VT_BBOX) &&
           VerifyField<uint16_t>(verifier, VT_CAT_ID) &&
           VerifyField<uint64_t>(verifier, VT_ID) &&
           VerifyField<uint64_t>(verifier, VT_IMAGE_ID) &&
           verifier.EndTable();
  }
};

struct AnnotationBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_bbox(const DLFS::BoundingBox *bbox) {
    fbb_.AddStruct(Annotation::VT_BBOX, bbox);
  }
  void add_cat_id(uint16_t cat_id) {
    fbb_.AddElement<uint16_t>(Annotation::VT_CAT_ID, cat_id, 0);
  }
  void add_id(uint64_t id) {
    fbb_.AddElement<uint64_t>(Annotation::VT_ID, id, 0);
  }
  void add_image_id(uint64_t image_id) {
    fbb_.AddElement<uint64_t>(Annotation::VT_IMAGE_ID, image_id, 0);
  }
  explicit AnnotationBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  AnnotationBuilder &operator=(const AnnotationBuilder &);
  flatbuffers::Offset<Annotation> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Annotation>(end);
    return o;
  }
};

inline flatbuffers::Offset<Annotation> CreateAnnotation(
    flatbuffers::FlatBufferBuilder &_fbb,
    const DLFS::BoundingBox *bbox = 0,
    uint16_t cat_id = 0,
    uint64_t id = 0,
    uint64_t image_id = 0) {
  AnnotationBuilder builder_(_fbb);
  builder_.add_image_id(image_id);
  builder_.add_id(id);
  builder_.add_bbox(bbox);
  builder_.add_cat_id(cat_id);
  return builder_.Finish();
}

struct Example FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_FILE_NAME = 4,
    VT_ID = 6,
    VT_ANNOTATIONS = 8,
    VT_IMAGE = 10,
    VT_WIDTH = 12,
    VT_HEIGHT = 14
  };
  const flatbuffers::String *file_name() const {
    return GetPointer<const flatbuffers::String *>(VT_FILE_NAME);
  }
  flatbuffers::String *mutable_file_name() {
    return GetPointer<flatbuffers::String *>(VT_FILE_NAME);
  }
  uint64_t id() const {
    return GetField<uint64_t>(VT_ID, 0);
  }
  bool mutate_id(uint64_t _id) {
    return SetField<uint64_t>(VT_ID, _id, 0);
  }
  bool KeyCompareLessThan(const Example *o) const {
    return id() < o->id();
  }
  int KeyCompareWithValue(uint64_t val) const {
    return static_cast<int>(id() > val) - static_cast<int>(id() < val);
  }
  const flatbuffers::Vector<flatbuffers::Offset<DLFS::Annotation>> *annotations() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<DLFS::Annotation>> *>(VT_ANNOTATIONS);
  }
  flatbuffers::Vector<flatbuffers::Offset<DLFS::Annotation>> *mutable_annotations() {
    return GetPointer<flatbuffers::Vector<flatbuffers::Offset<DLFS::Annotation>> *>(VT_ANNOTATIONS);
  }
  const flatbuffers::Vector<uint8_t> *image() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_IMAGE);
  }
  flatbuffers::Vector<uint8_t> *mutable_image() {
    return GetPointer<flatbuffers::Vector<uint8_t> *>(VT_IMAGE);
  }
  uint64_t width() const {
    return GetField<uint64_t>(VT_WIDTH, 0);
  }
  bool mutate_width(uint64_t _width) {
    return SetField<uint64_t>(VT_WIDTH, _width, 0);
  }
  uint64_t height() const {
    return GetField<uint64_t>(VT_HEIGHT, 0);
  }
  bool mutate_height(uint64_t _height) {
    return SetField<uint64_t>(VT_HEIGHT, _height, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_FILE_NAME) &&
           verifier.VerifyString(file_name()) &&
           VerifyField<uint64_t>(verifier, VT_ID) &&
           VerifyOffset(verifier, VT_ANNOTATIONS) &&
           verifier.VerifyVector(annotations()) &&
           verifier.VerifyVectorOfTables(annotations()) &&
           VerifyOffset(verifier, VT_IMAGE) &&
           verifier.VerifyVector(image()) &&
           VerifyField<uint64_t>(verifier, VT_WIDTH) &&
           VerifyField<uint64_t>(verifier, VT_HEIGHT) &&
           verifier.EndTable();
  }
};

struct ExampleBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_file_name(flatbuffers::Offset<flatbuffers::String> file_name) {
    fbb_.AddOffset(Example::VT_FILE_NAME, file_name);
  }
  void add_id(uint64_t id) {
    fbb_.AddElement<uint64_t>(Example::VT_ID, id, 0);
  }
  void add_annotations(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<DLFS::Annotation>>> annotations) {
    fbb_.AddOffset(Example::VT_ANNOTATIONS, annotations);
  }
  void add_image(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> image) {
    fbb_.AddOffset(Example::VT_IMAGE, image);
  }
  void add_width(uint64_t width) {
    fbb_.AddElement<uint64_t>(Example::VT_WIDTH, width, 0);
  }
  void add_height(uint64_t height) {
    fbb_.AddElement<uint64_t>(Example::VT_HEIGHT, height, 0);
  }
  explicit ExampleBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ExampleBuilder &operator=(const ExampleBuilder &);
  flatbuffers::Offset<Example> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Example>(end);
    return o;
  }
};

inline flatbuffers::Offset<Example> CreateExample(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> file_name = 0,
    uint64_t id = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<DLFS::Annotation>>> annotations = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> image = 0,
    uint64_t width = 0,
    uint64_t height = 0) {
  ExampleBuilder builder_(_fbb);
  builder_.add_height(height);
  builder_.add_width(width);
  builder_.add_id(id);
  builder_.add_image(image);
  builder_.add_annotations(annotations);
  builder_.add_file_name(file_name);
  return builder_.Finish();
}

inline flatbuffers::Offset<Example> CreateExampleDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *file_name = nullptr,
    uint64_t id = 0,
    const std::vector<flatbuffers::Offset<DLFS::Annotation>> *annotations = nullptr,
    const std::vector<uint8_t> *image = nullptr,
    uint64_t width = 0,
    uint64_t height = 0) {
  auto file_name__ = file_name ? _fbb.CreateString(file_name) : 0;
  auto annotations__ = annotations ? _fbb.CreateVector<flatbuffers::Offset<DLFS::Annotation>>(*annotations) : 0;
  auto image__ = image ? _fbb.CreateVector<uint8_t>(*image) : 0;
  return DLFS::CreateExample(
      _fbb,
      file_name__,
      id,
      annotations__,
      image__,
      width,
      height);
}

struct Dataset FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ID = 4,
    VT_EXAMPLES = 6,
    VT_CATEGORIES = 8
  };
  uint64_t id() const {
    return GetField<uint64_t>(VT_ID, 0);
  }
  bool mutate_id(uint64_t _id) {
    return SetField<uint64_t>(VT_ID, _id, 0);
  }
  const flatbuffers::Vector<flatbuffers::Offset<DLFS::Example>> *examples() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<DLFS::Example>> *>(VT_EXAMPLES);
  }
  flatbuffers::Vector<flatbuffers::Offset<DLFS::Example>> *mutable_examples() {
    return GetPointer<flatbuffers::Vector<flatbuffers::Offset<DLFS::Example>> *>(VT_EXAMPLES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<DLFS::Category>> *categories() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<DLFS::Category>> *>(VT_CATEGORIES);
  }
  flatbuffers::Vector<flatbuffers::Offset<DLFS::Category>> *mutable_categories() {
    return GetPointer<flatbuffers::Vector<flatbuffers::Offset<DLFS::Category>> *>(VT_CATEGORIES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_ID) &&
           VerifyOffset(verifier, VT_EXAMPLES) &&
           verifier.VerifyVector(examples()) &&
           verifier.VerifyVectorOfTables(examples()) &&
           VerifyOffset(verifier, VT_CATEGORIES) &&
           verifier.VerifyVector(categories()) &&
           verifier.VerifyVectorOfTables(categories()) &&
           verifier.EndTable();
  }
};

struct DatasetBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_id(uint64_t id) {
    fbb_.AddElement<uint64_t>(Dataset::VT_ID, id, 0);
  }
  void add_examples(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<DLFS::Example>>> examples) {
    fbb_.AddOffset(Dataset::VT_EXAMPLES, examples);
  }
  void add_categories(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<DLFS::Category>>> categories) {
    fbb_.AddOffset(Dataset::VT_CATEGORIES, categories);
  }
  explicit DatasetBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DatasetBuilder &operator=(const DatasetBuilder &);
  flatbuffers::Offset<Dataset> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Dataset>(end);
    return o;
  }
};

inline flatbuffers::Offset<Dataset> CreateDataset(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t id = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<DLFS::Example>>> examples = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<DLFS::Category>>> categories = 0) {
  DatasetBuilder builder_(_fbb);
  builder_.add_id(id);
  builder_.add_categories(categories);
  builder_.add_examples(examples);
  return builder_.Finish();
}

inline flatbuffers::Offset<Dataset> CreateDatasetDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t id = 0,
    const std::vector<flatbuffers::Offset<DLFS::Example>> *examples = nullptr,
    const std::vector<flatbuffers::Offset<DLFS::Category>> *categories = nullptr) {
  auto examples__ = examples ? _fbb.CreateVector<flatbuffers::Offset<DLFS::Example>>(*examples) : 0;
  auto categories__ = categories ? _fbb.CreateVector<flatbuffers::Offset<DLFS::Category>>(*categories) : 0;
  return DLFS::CreateDataset(
      _fbb,
      id,
      examples__,
      categories__);
}

inline const DLFS::Dataset *GetDataset(const void *buf) {
  return flatbuffers::GetRoot<DLFS::Dataset>(buf);
}

inline const DLFS::Dataset *GetSizePrefixedDataset(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<DLFS::Dataset>(buf);
}

inline Dataset *GetMutableDataset(void *buf) {
  return flatbuffers::GetMutableRoot<Dataset>(buf);
}

inline bool VerifyDatasetBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<DLFS::Dataset>(nullptr);
}

inline bool VerifySizePrefixedDatasetBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<DLFS::Dataset>(nullptr);
}

inline void FinishDatasetBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<DLFS::Dataset> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedDatasetBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<DLFS::Dataset> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace DLFS

#endif  // FLATBUFFERS_GENERATED_DATASET_DLFS_H_
