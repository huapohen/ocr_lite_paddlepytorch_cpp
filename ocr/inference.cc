#include "inference.h"

#include "inference_impl.h"

namespace ocr {
Inference::Inference() = default;

Inference::~Inference() = default;

InferencePtr Inference::CreateInstance() {
  return std::make_shared<InferenceImpl>();
}
}  // namespace ocr
