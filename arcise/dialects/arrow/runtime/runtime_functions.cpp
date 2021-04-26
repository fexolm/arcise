#include <cstddef>
#include <cstdint>
extern "C" {

struct RecordBatch {
  size_t length;
  int8_t **arrays;
};

struct MemRef {
  int8_t *data;
};

struct Array {
  MemRef null_bitmap;
  MemRef data_buffer;
  size_t length;
};

Array *get_column(RecordBatch *rb, size_t index) {
  return reinterpret_cast<Array *>(rb->arrays[index]);
}

MemRef get_data_buffer(Array *array) { return array->data_buffer; }

MemRef get_null_bitmap(Array *array) { return array->null_bitmap; }

size_t get_length(Array *array) { return array->length; }

size_t get_rows_count(RecordBatch *record_batch) {
  return record_batch->length;
}

Array *make_array(MemRef null_bitmap, MemRef data_buffer, size_t len) {
  return new Array{null_bitmap, data_buffer, len};
}

RecordBatch *make_record_batch(size_t length) {
  return new RecordBatch{length, new int8_t *[length]};
}

RecordBatch *get_record_batch(char *name) { return nullptr; }
}
