#ifndef INFERENCE_IMPL_H
#define INFERENCE_IMPL_H

#include "cls.h"
#include "det.h"
#include "inference.h"
#include "rec.h"

namespace ocr {

class InferenceImpl : public Inference {
 public:
  InferenceImpl();
  virtual ~InferenceImpl();
  void initial(std::string bin_dir_path, std::string config_dir_path,
               int input_width, int input_height) override;
  void inference(unsigned char* yuv_addr) override;
  std::vector<std::string> GetResult() override;

 private:
  cv::Mat get_input(unsigned char* yuv_addr, cv::Mat& img_bgr);
  bool read_config_json(std::string config_json_path);
  std::vector<OCRPredictResult> ocr_results;

  bool m_non_bev_test_mode = false;
  bool m_one_step_remap_mode = false;
  bool m_multiply_float_model = false;
  bool m_take_first_string = true;

  Det det_handle;
  Cls cls_handle;
  Rec rec_handle;

  Util util_handle;

  bool is_null_json = false;
  bool is_null_label = false;
  bool is_null_map_bin = false;

  cv::Mat img_src;
  cv::Mat img_bev;
  cv::Mat img_det_gray;
  cv::Mat img_det_bgr;
  cv::Mat img_fev_resize;
  cv::Mat img_bev_amplify;

  cv::Mat m_map_x;
  cv::Mat m_map_y;
  int m_input_height;
  int m_input_width;
  int m_det_inp_h;
  int m_det_inp_w;
  int m_bev_h;
  int m_bev_w;
  int m_fev_resize_h;
  int m_fev_resize_w;
  // hyper-parameter
  int m_dh_shift;
  int m_dw_shift;
  float m_roi_scale;
};
}  // namespace ocr
#endif