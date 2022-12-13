#ifndef DET_H
#define DET_H

#include "tool/util.h"

namespace ocr {

class Det {
 public:
  Det();
  virtual ~Det();
  void initial(std::string config_dir_path, std::string config_json_path);
  void preprocess(cv::Mat srcimg);
  void inference();
  void postprocess(std::vector<OCRPredictResult>& ocr_results);

  vvveci boxes;
  ncnn::Mat net_inp_data;
  ncnn::Mat net_out_data;

 private:
  Util util_handle;
  // post-process
  DBPostProcessor post_processor_;

  float ratio_h;
  float ratio_w;

  double det_db_thresh_;
  double det_db_box_thresh_;
  double det_db_unclip_ratio_;
  std::string det_db_score_mode_;
  bool use_dilation_;

  float img_h;
  float img_w;

  std::vector<float> mean_value;
  std::vector<float> std_value;

  ncnn::Net net;
};
}  // namespace ocr
#endif