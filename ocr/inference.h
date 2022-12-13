#ifndef INFERENCE_H
#define INFERENCE_H

#include <memory>
#include <string>
#include <vector>

namespace ocr {

class Inference;
using InferencePtr = std::shared_ptr<Inference>;

class Inference {
 public:
  Inference();
  virtual ~Inference();
  static InferencePtr CreateInstance();
  virtual void initial(std::string bin_dir_path, std::string config_dir_path,
                       int input_width, int input_height) = 0;
  virtual void inference(unsigned char* yuv_addr) = 0;
  virtual std::vector<std::string> GetResult() = 0;
};
}  // namespace ocr
#endif