#include "cls.h"

#include <algorithm>
namespace ocr {
Cls::Cls() = default;

Cls::~Cls() = default;

void Cls::initial(std::string config_dir_path, std::string config_json_path) {
  Json::Value value;
  Json::Reader reader;
  std::ifstream ifs(config_json_path);
  reader.parse(ifs, value);

  // cls init
  is_scale = value["cls"]["is_scale"].asBool();
  cls_batch_num_ = value["cls"]["batch_num"].asInt();
  cls_thresh = value["cls"]["thresh"].asFloat();
  const Json::Value mean_arr = value["cls"]["mean_value"];
  mean_value = {mean_arr[0].asFloat(), mean_arr[1].asFloat(),
                mean_arr[2].asFloat()};
  const Json::Value std_arr = value["cls"]["std_value"];
  std_value = {std_arr[0].asFloat(), std_arr[1].asFloat(),
               std_arr[2].asFloat()};
  const Json::Value image_shape_arr = value["cls"]["image_shape"];
  cls_image_shape = {image_shape_arr[0].asInt(), image_shape_arr[1].asInt(),
                     image_shape_arr[2].asInt()};

  // ncnn init
  bool lightmode = value["cls"]["ncnn_lightmode"].asBool();
  int num_thread = value["ncnn_num_thread"].asInt();
  int cpu_powersave = value["ncnn_cpu_powersave"].asInt();
  int omp_dynamic = value["ncnn_omp_dynamic"].asInt();
  std::string model_bin_str =
      config_dir_path + value["cls"]["model_bin_path"].asString();
  std::string model_param_str =
      config_dir_path + value["cls"]["model_param_path"].asString();

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

void Cls::preprocess(cv::Mat srcimg) {
  img_list.clear();
  cls_labels.clear();
  cls_scores.clear();
  // crop subimg from origin img
  for (unsigned int i = 0; i < ocr_results.size(); i++) {
    cv::Mat crop_img;
    crop_img = Utility::GetRotateCropImage(srcimg, ocr_results[i].box);
    img_list.push_back(crop_img);
  }
  // assign size
  for (unsigned int i = 0; i < img_list.size(); i++) {
    cls_labels.push_back(0);
    cls_scores.push_back(0.f);
  }
}

void Cls::inference() {
  int img_num = img_list.size();

  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += this->cls_batch_num_) {
    int end_img_no = std::min(img_num, beg_img_no + this->cls_batch_num_);
    predict_batch.clear();
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      // preprocess
      cv::Mat t_img;
      img_list[ino].copyTo(t_img);
      cv::Mat resize_img = util_handle.cls_resize(t_img, cls_image_shape);
      cv::Mat norm_img =
          util_handle.normalize(resize_img, mean_value, std_value, is_scale);
      cv::cvtColor(norm_img, norm_img, cv::COLOR_BGR2RGB);

      if (resize_img.cols < cls_image_shape[2]) {
        cv::copyMakeBorder(norm_img, norm_img, 0, 0, 0,
                           cls_image_shape[2] - norm_img.cols,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      }
      ncnn::Mat in_pack3(norm_img.cols, norm_img.rows, 1, (void*)norm_img.data,
                         (size_t)4u * 3, 3);

      ncnn::Mat sub_inp_data, sub_out_data;
      ncnn::convert_packing(in_pack3, sub_inp_data, 1);

      const std::vector<const char*>& input_names = net.input_names();
      const std::vector<const char*>& output_names = net.output_names();

      ncnn::Extractor net_extractor = net.create_extractor();

      // inference
      net_extractor.input(input_names[0], sub_inp_data);
      net_extractor.extract(output_names[0], sub_out_data);

      int h = sub_out_data.h;  //
      int w = sub_out_data.w;  //
      int c = sub_out_data.c;  //

      int count = h * w * c;
      sub_out_data.reshape(count);
      std::vector<float> out_data(count);
      for (int k = 0; k < count; k++) {
        out_data[k] = sub_out_data[k];
      }

      predict_batch.push_back(out_data);
    }

    // postprocess
    for (unsigned int batch_idx = 0; batch_idx < predict_batch.size();
         batch_idx++) {
      auto pred_data = predict_batch[batch_idx];
      auto iter = std::max_element(pred_data.begin(), pred_data.end());
      auto index = std::distance(pred_data.begin(), iter);
      float value = *iter;
      cls_labels[beg_img_no + batch_idx] = index;
      cls_scores[beg_img_no + batch_idx] = value;
    }
  }
}

void Cls::postprocess() {
  // output cls results
  for (unsigned int i = 0; i < cls_labels.size(); i++) {
    ocr_results[i].cls_label = cls_labels[i];
    ocr_results[i].cls_score = cls_scores[i];
  }
  // rotate cropped subimg
  for (unsigned int i = 0; i < img_list.size(); i++) {
    if (ocr_results[i].cls_label % 2 == 1 &&
        ocr_results[i].cls_score > cls_thresh) {
      cv::rotate(img_list[i], img_list[i], 1);
    }
  }
}

}  // namespace ocr