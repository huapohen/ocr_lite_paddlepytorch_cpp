#ifndef REC_H
#define REC_H

#include "tool/util.h"

namespace ocr {

class Rec {
 public:
  Rec();
  virtual ~Rec();
  void initial(std::string config_dir_path, std::string config_json_path);
  void preprocess();
  void inference();
  void postprocess();

  bool is_null_label = false;

  std::vector<cv::Mat> img_list;
  std::vector<OCRPredictResult> ocr_results;

 private:
  Util util_handle;

  std::vector<std::string> rec_texts;
  std::vector<float> rec_text_scores;
  std::vector<float> width_list;
  std::vector<int> indices;
  std::vector<std::vector<float>> predict_batch;

  std::vector<std::string> label_list_;
  std::string label_path;

  int rec_batch_num_;
  std::vector<int> rec_image_shape_;

  std::vector<float> mean_value;
  std::vector<float> std_value;
  bool is_scale;

  ncnn::Net net;
};
}  // namespace ocr
#endif