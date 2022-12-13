#include "rec.h"

namespace ocr {
Rec::Rec() = default;

Rec::~Rec() = default;

void Rec::initial(std::string config_dir_path, std::string config_json_path) {
  Json::Value value;
  Json::Reader reader;
  std::ifstream ifs(config_json_path);
  reader.parse(ifs, value);

  // rec init
  is_scale = value["rec"]["is_scale"].asBool();
  rec_batch_num_ = value["rec"]["batch_num"].asInt();
  const Json::Value mean_arr = value["rec"]["mean_value"];
  mean_value = {mean_arr[0].asFloat(), mean_arr[1].asFloat(),
                mean_arr[2].asFloat()};
  const Json::Value std_arr = value["rec"]["std_value"];
  std_value = {std_arr[0].asFloat(), std_arr[1].asFloat(),
               std_arr[2].asFloat()};
  const Json::Value image_shape_arr = value["rec"]["image_shape"];
  rec_image_shape_ = {image_shape_arr[0].asInt(), image_shape_arr[1].asInt(),
                      image_shape_arr[2].asInt()};
  // character
  label_path = config_dir_path + value["rec"]["label_path"].asString();
  this->label_list_ = Utility::ReadDict(label_path);
  if ((int)this->label_list_.size() == 0) {
    is_null_label = true;
    return;
  }
  this->label_list_.insert(this->label_list_.begin(),
                           "#");  // blank char for ctc
  this->label_list_.push_back(" ");

  // ncnn init
  bool lightmode = value["rec"]["ncnn_lightmode"].asBool();
  int num_thread = value["ncnn_num_thread"].asInt();
  int cpu_powersave = value["ncnn_cpu_powersave"].asInt();
  int omp_dynamic = value["ncnn_omp_dynamic"].asInt();
  std::string model_bin_str =
      config_dir_path + value["rec"]["model_bin_path"].asString();
  std::string model_param_str =
      config_dir_path + value["rec"]["model_param_path"].asString();

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

void Rec::preprocess() {
  // assign size
  rec_texts.clear();
  rec_text_scores.clear();
  width_list.clear();
  indices.clear();
  for (unsigned int i = 0; i < img_list.size(); i++) {
    rec_texts.push_back("");
    rec_text_scores.push_back(0.f);
    width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
  }
  indices = Utility::argsort(width_list);
}

void Rec::inference() {
  int img_num = img_list.size();

  int out_c, out_h, out_w;
  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += this->rec_batch_num_) {
    int end_img_no = std::min(img_num, beg_img_no + this->rec_batch_num_);
    int imgH = this->rec_image_shape_[1];
    int imgW = this->rec_image_shape_[2];
    float max_wh_ratio = imgW * 1.0 / imgH;
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      int h = img_list[indices[ino]].rows;
      int w = img_list[indices[ino]].cols;
      float wh_ratio = w * 1.0 / h;
      max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
    }

    // predict_batch is the result of Last FC with softmax
    predict_batch.clear();
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      cv::Mat srcimg;
      img_list[indices[ino]].copyTo(srcimg);
      cv::Mat resize_img =
          util_handle.crnn_resize(srcimg, max_wh_ratio, this->rec_image_shape_);
      cv::Mat norm_img =
          util_handle.normalize(resize_img, mean_value, std_value, is_scale);
      cv::Mat rgb_img;
      cv::cvtColor(norm_img, rgb_img, cv::COLOR_BGR2RGB);

      ncnn::Mat in_pack3(rgb_img.cols, rgb_img.rows, 1, (void*)rgb_img.data,
                         (size_t)4u * 3, 3);

      ncnn::Mat sub_inp_data, sub_out_data;
      ncnn::convert_packing(in_pack3, sub_inp_data, 1);

      ncnn::Extractor net_extractor = net.create_extractor();

      const std::vector<const char*>& input_names = net.input_names();
      const std::vector<const char*>& output_names = net.output_names();
      // inference
      int out_idx = output_names.size() - 1;
      net_extractor.input(input_names[0], sub_inp_data);
      net_extractor.extract(output_names[out_idx], sub_out_data);

      out_h = sub_out_data.h;  // 80
      out_w = sub_out_data.w;  // 6625 -> 5531
      out_c = sub_out_data.c;  // 1

      int count = out_h * out_w * out_c;
      sub_out_data.reshape(count);
      std::vector<float> out_data(count);
      for (int k = 0; k < count; k++) {
        out_data[k] = sub_out_data[k];
      }

      predict_batch.push_back(out_data);
    }

    // ctc decode
    for (unsigned int batch_idx = 0; batch_idx < predict_batch.size();
         batch_idx++) {
      std::string str_res;
      int last_index = 0;
      float score = 0.f;
      int count = 0;
      auto pred_data = predict_batch[batch_idx];
      for (int hi = 0; hi < out_h; hi++) {
        int index = int(Utility::argmax(&pred_data[hi * out_w],
                                        &pred_data[hi * out_w + out_w]));
        float value = float(*std::max_element(&pred_data[hi * out_w],
                                              &pred_data[hi * out_w + out_w]));
        if (index > 0 && (!(hi > 0 && index == last_index))) {
          score += value;
          count += 1;
          str_res += label_list_[index];
        }
        last_index = index;
      }
      score /= count;
      if (std::isnan(score)) {
        continue;
      }
      rec_texts[indices[beg_img_no + batch_idx]] = str_res;
      rec_text_scores[indices[beg_img_no + batch_idx]] = score;
    }
  }
}

void Rec::postprocess() {
  // output rec results
  for (unsigned int i = 0; i < rec_texts.size(); i++) {
    ocr_results[i].text = rec_texts[i];
    ocr_results[i].score = rec_text_scores[i];
  }
}

}  // namespace ocr