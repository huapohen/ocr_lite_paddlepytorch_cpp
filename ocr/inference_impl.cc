#include "inference_impl.h"

namespace ocr {
InferenceImpl::InferenceImpl() = default;

InferenceImpl::~InferenceImpl() = default;

void InferenceImpl::initial(std::string bin_dir_path,
                            std::string config_dir_path, int input_width,
                            int input_height) {
  auto time_start = std::chrono::steady_clock::now();

  m_input_width = input_width;
  m_input_height = input_height;

  std::string config_json_path = config_dir_path + "ocr_config.json";
  std::string bin_path = bin_dir_path + "god_center_rear_bin";

  bool valid_json = read_config_json(config_json_path);
  if (!valid_json) {
    is_null_json = true;
    return;
  }

  det_handle.initial(config_dir_path, config_json_path);
  cls_handle.initial(config_dir_path, config_json_path);
  rec_handle.initial(config_dir_path, config_json_path);

  if (rec_handle.is_null_label) {
    is_null_label = true;
    return;
  }

  if (m_one_step_remap_mode) {
    m_map_x = cv::Mat(m_det_inp_h, m_det_inp_w, CV_32F);
    m_map_y = cv::Mat(m_det_inp_h, m_det_inp_w, CV_32F);
    img_bev = cv::Mat(m_det_inp_h, m_det_inp_w, CV_8UC1);
  } else {
    m_map_x = cv::Mat(m_bev_h, m_bev_w, CV_32F);
    m_map_y = cv::Mat(m_bev_h, m_bev_w, CV_32F);
    img_bev = cv::Mat(m_bev_h, m_bev_w, CV_8UC1);
    img_bev_amplify =
        cv::Mat(m_bev_h * m_roi_scale, m_bev_w * m_roi_scale, CV_8UC1);
  }
  bool is_open = util_handle.read_bin_offset(
      m_one_step_remap_mode, m_multiply_float_model, bin_path, m_map_x, m_map_y,
      m_det_inp_h, m_det_inp_w, m_input_height, m_input_width, m_dh_shift,
      m_dw_shift, m_roi_scale);
  if (is_open == false) {
    is_null_map_bin = true;
    return;
  }

  img_src = cv::Mat(m_input_height, m_input_width, CV_8UC1);
  img_det_gray = cv::Mat(m_det_inp_h, m_det_inp_w, CV_8UC1);
  img_det_bgr = cv::Mat(m_det_inp_h, m_det_inp_w, CV_8UC3);
  img_fev_resize = cv::Mat(m_fev_resize_h, m_fev_resize_w, CV_8UC3);

  auto time_end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time_diff = time_end - time_start;
  std::cout << "all initial cost: " << int(time_diff.count() * 1000)
            << std::endl;
}

bool InferenceImpl::read_config_json(std::string config_json_path) {
  Json::Value ocr_cfg;
  Json::Reader reader;
  std::ifstream ifs_ocr(config_json_path);
  if (!reader.parse(ifs_ocr, ocr_cfg)) {
    std::cout << "fail to parse ocr config json, path:" << config_json_path
              << std::endl;
    return false;
  }
  m_non_bev_test_mode = ocr_cfg["non_bev_test_mode"].asBool();
  m_one_step_remap_mode = ocr_cfg["one_step_remap_mode"].asBool();
  m_multiply_float_model = ocr_cfg["multiply_float_model"].asBool();
  m_take_first_string = ocr_cfg["take_first_string"].asBool();

  m_det_inp_h = ocr_cfg["det_input_h"].asInt();
  m_det_inp_w = ocr_cfg["det_input_w"].asInt();
  m_bev_h = ocr_cfg["bev_h"].asInt();
  m_bev_w = ocr_cfg["bev_w"].asInt();
  m_fev_resize_h = ocr_cfg["fev_resize_h"].asInt();
  m_fev_resize_w = ocr_cfg["fev_resize_w"].asInt();
  m_dh_shift = ocr_cfg["dh_shift"].asInt();
  m_dw_shift = ocr_cfg["dw_shift"].asInt();
  m_roi_scale = ocr_cfg["roi_scale"].asFloat();

  return true;
}

cv::Mat InferenceImpl::get_input(unsigned char* yuv_addr, cv::Mat& img_bgr) {
  auto time_start = std::chrono::steady_clock::now();

  img_src = cv::Mat(m_input_height, m_input_width, CV_8UC1, yuv_addr);

  if (m_non_bev_test_mode) {
    cv::resize(img_src, img_fev_resize,
               cv::Size(m_fev_resize_w, m_fev_resize_h));
    int dw = (m_fev_resize_w - m_det_inp_w) / 2;
    int dh = (m_fev_resize_h - m_det_inp_h) / 2;
    img_det_gray =
        img_fev_resize(cv::Rect(dw, dh, m_det_inp_w, m_det_inp_h)).clone();
    cv::cvtColor(img_det_gray, img_det_bgr, CV_GRAY2BGR);
    auto time_end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_diff = time_end - time_start;
    std::cout << "\nget input img cost: " << int(time_diff.count() * 1000)
              << std::endl;  // milliseconds
    return img_det_bgr;
  }

  if (m_one_step_remap_mode) {
    img_src = cv::Mat(m_input_height, m_input_width, CV_8UC1, yuv_addr);
    cv::remap(img_src, img_bev, m_map_x, m_map_y, cv::INTER_LINEAR);
    cv::cvtColor(img_bev, img_det_bgr, CV_GRAY2BGR);
  } else {
    img_src = cv::Mat(m_input_height, m_input_width, CV_8UC1, yuv_addr);
    cv::resize(img_src, img_fev_resize,
               cv::Size(m_fev_resize_w, m_fev_resize_h));
    cv::remap(img_fev_resize, img_bev, m_map_x, m_map_y, cv::INTER_LINEAR);
    cv::flip(img_bev, img_bev, -1);
    cv::resize(
        img_bev, img_bev_amplify,
        cv::Size(img_bev.cols * m_roi_scale, img_bev.rows * m_roi_scale));
    int dh = img_bev_amplify.rows - m_det_inp_h - m_dh_shift;
    int dw = (img_bev_amplify.cols - m_det_inp_w) / 2 - m_dw_shift;

    img_det_gray =
        img_bev_amplify(cv::Rect(dw, dh, m_det_inp_w, m_det_inp_h)).clone();
    cv::cvtColor(img_det_gray, img_det_bgr, CV_GRAY2BGR);
  }

  // cv::imwrite("result/bev_crop.jpg", img_det_bgr);

  auto time_end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time_diff = time_end - time_start;
  std::cout << "\nget input img cost: " << int(time_diff.count() * 1000)
            << std::endl;  // milliseconds
  return img_det_bgr;
}

void InferenceImpl::inference(unsigned char* yuv_addr) {
  if (is_null_json || is_null_label || is_null_map_bin) {
    return;
  }

  ocr_results.clear();

  /* ========================= Start ========================= */
  auto time_all_start = std::chrono::steady_clock::now();

  // get input fev image, and convert it to bev image
  get_input(yuv_addr, img_det_bgr);
  /* ========================= det ========================= */
  auto time_det_start = std::chrono::steady_clock::now();

  this->det_handle.preprocess(img_det_bgr);
  this->det_handle.inference();
  this->det_handle.postprocess(ocr_results);

  auto time_det_end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time_det_diff = time_det_end - time_det_start;
  std::cout << "det cost: " << int(time_det_diff.count() * 1000) << std::endl;

  // Utility::print_result(ocr_results);
  // Utility::VisualizeBboxes(img_src, ocr_results, "result/ocr_box.jpg");

  /* ========================= cls ========================= */
  auto time_cls_start = std::chrono::steady_clock::now();

  this->cls_handle.ocr_results = ocr_results;
  this->cls_handle.preprocess(img_det_bgr);
  this->cls_handle.inference();
  this->cls_handle.postprocess();
  ocr_results = this->cls_handle.ocr_results;
  auto subimg_list = this->cls_handle.img_list;

  auto time_cls_end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time_cls_diff = time_cls_end - time_cls_start;
  std::cout << "cls cost: " << int(time_cls_diff.count() * 1000) << std::endl;

  /* ========================= rec ========================= */
  auto time_rec_start = std::chrono::steady_clock::now();

  this->rec_handle.ocr_results = ocr_results;
  this->rec_handle.img_list = subimg_list;
  this->rec_handle.preprocess();
  this->rec_handle.inference();
  this->rec_handle.postprocess();
  ocr_results = this->rec_handle.ocr_results;

  auto time_rec_end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time_rec_diff = time_rec_end - time_rec_start;
  std::cout << "rec cost: " << int(time_rec_diff.count() * 1000) << std::endl;

  // Utility::print_result(ocr_results);
  // Utility::VisualizeBboxes(img_src, ocr_results, "result/ocr_box.jpg");

  /* ========================= The End ========================= */
  auto time_all_end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time_all_diff = time_all_end - time_all_start;
  std::cout << "all inference cost: " << int(time_all_diff.count() * 1000)
            << std::endl;
  std::cout << std::endl;
}

std::vector<std::string> InferenceImpl::GetResult() {
  std::vector<std::string> result, str_tmp;
  for (auto& predict : ocr_results) {
    int text_length = predict.text.size();
    // if (false) {
    if (true) { // only output 0~9 and a~z(A~Z)
      // A01/001, A001/0001
      if (text_length > 2 && text_length < 5) {
        // filter
        char ch = predict.text[0];
        if ((ch - 'A' >= 0 && ch - 'Z' <= 0) ||
            (ch - 'a' >= 0 && ch - 'z' <= 0) ||
            (ch - '0' >= 0 && ch - '9' <= 0)) {
          bool valid = true;
          for (int i = 1; i < text_length; i++) {
            char c = predict.text[i];
            if (!(c - '0' >= 0 && c - '9' <= 0)) {
              valid = false;
              break;
            }
          }
          if (valid) {
            str_tmp.push_back(predict.text);
          }
        }
      }
    } else { // output all 5532 characters
      str_tmp.push_back(predict.text);
    }
  }
  if ((int)str_tmp.size() > 0) {
    if (m_take_first_string) { // only output first string
      result.push_back(str_tmp[0]);
    } else { // output all ocr results
      result.assign(str_tmp.begin(), str_tmp.end());
    }
  }

  return result;
}

}  // namespace ocr
