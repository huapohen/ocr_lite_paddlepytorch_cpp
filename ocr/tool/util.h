#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#include <opencv2/imgproc/types_c.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "datatype.h"
#include "jsoncpp/json/json.h"
#include "ncnn/cpu.h"
#include "ncnn/mat.h"
#include "ncnn/net.h"
#include "postprocess_op.h"
#include "utility.h"

namespace ocr {

class Util {
 public:
  void binMatWrite(cv::Mat map_x, cv::Mat map_y, FILE *fpData);
  void matrix_mul_3x3(cv::Mat H, float j, float i, cv::Vec2f &label,
                      bool multiply_float_model);
  bool read_bin_offset(bool one_step_remap_mode, bool multiply_float_model,
                       const std::string &bin_path, cv::Mat &map_x,
                       cv::Mat &map_y, int det_inp_h, int det_inp_w,
                       int input_height, int input_width, int dh_shift,
                       int dw_shift, float roi_scale);
  cv::Mat det_wh_check_resize(const cv::Mat img, float &ratio_h,
                              float &ratio_w);
  cv::Mat normalize(cv::Mat srcimg, const std::vector<float> &mean_value,
                    const std::vector<float> &std_value, const bool is_scale);
  cv::Mat cls_resize(const cv::Mat &img,
                     const std::vector<int> &rec_image_shape);
  cv::Mat crnn_resize(const cv::Mat &img, float wh_ratio,
                      const std::vector<int> &rec_image_shape);

 private:
  std::string limit_type = "max";
  int limit_side_len = 960;
};
}  // namespace ocr
#endif