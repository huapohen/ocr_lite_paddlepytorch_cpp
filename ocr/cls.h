#ifndef CLS_H
#define CLS_H

#include "tool/util.h"

namespace ocr {

class Cls {
 public:
  Cls();
  virtual ~Cls();
  void initial(std::string config_dir_path, std::string config_json_path);
  void preprocess(cv::Mat srcimg);
  void inference();
  void postprocess();

  std::vector<cv::Mat> img_list;
  std::vector<OCRPredictResult> ocr_results;

 private:
  Util util_handle;
  std::vector<int> cls_labels;
  std::vector<float> cls_scores;
  std::vector<std::vector<float>> predict_batch;

  std::vector<int> cls_image_shape;
  std::vector<float> mean_value;
  std::vector<float> std_value;
  bool is_scale;
  int cls_batch_num_;
  float cls_thresh;

  ncnn::Net net;
};
}  // namespace ocr
#endif