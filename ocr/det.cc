#include "det.h"

namespace ocr {
Det::Det() = default;

Det::~Det() = default;

void Det::initial(std::string config_dir_path, std::string config_json_path) {
  Json::Value value;
  Json::Reader reader;
  std::ifstream ifs(config_json_path);
  reader.parse(ifs, value);

  // det init
  det_db_thresh_ = value["det"]["db_thresh"].asFloat();
  det_db_box_thresh_ = value["det"]["db_box_thresh"].asFloat();
  det_db_unclip_ratio_ = value["det"]["db_unclip_ratio"].asFloat();
  det_db_score_mode_ = value["det"]["db_score_mode"].asString();
  use_dilation_ = value["det"]["use_dilation"].asBool();
  const Json::Value mean_arr = value["det"]["mean_value"];
  mean_value = {mean_arr[0].asFloat(), mean_arr[1].asFloat(),
                mean_arr[2].asFloat()};
  const Json::Value std_arr = value["det"]["std_value"];
  std_value = {std_arr[0].asFloat(), std_arr[1].asFloat(),
               std_arr[2].asFloat()};

  // ncnn init
  bool lightmode = value["det"]["ncnn_lightmode"].asBool();
  int num_thread = value["ncnn_num_thread"].asInt();
  int cpu_powersave = value["ncnn_cpu_powersave"].asInt();
  int omp_dynamic = value["ncnn_omp_dynamic"].asInt();
  std::string model_bin_str =
      config_dir_path + value["det"]["model_bin_path"].asString();
  std::string model_param_str =
      config_dir_path + value["det"]["model_param_path"].asString();

  ncnn::set_omp_num_threads(num_thread);
  ncnn::set_cpu_powersave(cpu_powersave);
  ncnn::set_omp_dynamic(omp_dynamic);

  ncnn::Option opt;
  opt.lightmode = lightmode;
  opt.num_threads = num_thread;
  net.opt = opt;

  // load model
  net.load_param(model_param_str.c_str());
  net.load_model(model_bin_str.c_str());
}

void Det::preprocess(cv::Mat srcimg) {
  auto preprocess_start = std::chrono::steady_clock::now();
  img_h = srcimg.rows;
  img_w = srcimg.cols;
  // resolution check: set h % 32 = 0 and w % 32 = 0
  cv::Mat _img = util_handle.det_wh_check_resize(srcimg, ratio_h, ratio_w);
  // normalize directly
  cv::Mat dst = util_handle.normalize(_img, mean_value, std_value, true);
  cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

  ncnn::Mat in_pack3(dst.cols, dst.rows, 1, (void*)dst.data, (size_t)4u * 3, 3);
  ncnn::convert_packing(in_pack3, this->net_inp_data, 1);

  auto preprocess_end = std::chrono::steady_clock::now();
  std::chrono::duration<float> preprocess_diff =
      preprocess_end - preprocess_start;
  std::cout << "[det] preprocess cost: " << int(preprocess_diff.count() * 1000)
            << std::endl;
}

void Det::inference() {
  auto inference_start = std::chrono::steady_clock::now();

  const std::vector<const char*>& input_names = net.input_names();
  const std::vector<const char*>& output_names = net.output_names();

  ncnn::Extractor net_extractor = net.create_extractor();

  net_extractor.input(input_names[0], this->net_inp_data);
  net_extractor.extract(output_names[0], this->net_out_data);

  auto inference_end = std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  std::cout << "[det] inference cost: " << int(inference_diff.count() * 1000)
            << std::endl;
}

void Det::postprocess(std::vector<OCRPredictResult>& ocr_results) {
  auto postprocess_start = std::chrono::steady_clock::now();

  int h = this->net_out_data.h;  // 160
  int w = this->net_out_data.w;  // 352
  int c = this->net_out_data.c;  // 1

  int count = h * w * c;
  this->net_out_data.reshape(count);
  std::vector<float> out_data(count);
  for (int i = 0; i < count; i++) {
    out_data[i] = this->net_out_data[i];
  }

  int n = h * w;
  std::vector<float> pred(n, 0.0);
  std::vector<unsigned char> cbuf(n, ' ');

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    cbuf[i] = (unsigned char)((out_data[i]) * 255);
  }

  cv::Mat cbuf_map(h, w, CV_8UC1, (unsigned char*)cbuf.data());
  cv::Mat pred_map(h, w, CV_32F, (float*)pred.data());

  const double threshold = this->det_db_thresh_ * 255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
  if (this->use_dilation_) {
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, bit_map, dila_ele);
  }

  this->boxes = post_processor_.BoxesFromBitmap(
      pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
      this->det_db_score_mode_);

  this->boxes = post_processor_.FilterTagDetRes(this->boxes, ratio_h, ratio_w,
                                                img_h, img_w);

  // sort out boxes
  for (unsigned int i = 0; i < boxes.size(); i++) {
    OCRPredictResult res;
    res.box = boxes[i];
    ocr_results.push_back(res);
  }
  // sort boex from top to bottom, from left to right
  Utility::sorted_boxes(ocr_results);

  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> postprocess_diff =
      postprocess_end - postprocess_start;
  std::cout << "[det] postprocess cost: "
            << int(postprocess_diff.count() * 1000) << std::endl;
}

}  // namespace ocr