#include "util.h"

namespace ocr {

void Util::matrix_mul_3x3(cv::Mat H, float j, float i, cv::Vec2f &label,
                          bool multiply_float_model) {
  if (!multiply_float_model) {
    j = int(j);
    i = int(i);
  }
  float div = H.at<float>(2, 0) * j + H.at<float>(2, 1) * i + H.at<float>(2, 2);
  label[0] =
      (H.at<float>(0, 0) * j + H.at<float>(0, 1) * i + H.at<float>(0, 2)) / div;
  label[1] =
      (H.at<float>(1, 0) * j + H.at<float>(1, 1) * i + H.at<float>(1, 2)) / div;
}

bool Util::read_bin_offset(bool one_step_remap_mode, bool multiply_float_model,
                           const std::string &bin_path, cv::Mat &map_x,
                           cv::Mat &map_y, int det_inp_h, int det_inp_w,
                           int input_height, int input_width, int dh_shift,
                           int dw_shift, float roi_scale) {
  std::ifstream inFile(bin_path, std::ios::in | std::ios::binary);
  if (!inFile.is_open()) {
    std::cout << "open file error: " << bin_path << std::endl;
    return false;
  }
  int *test = new int[2];
  inFile.read((char *)test, sizeof(int) * 2);
  int h = test[1];  // 192
  int w = test[0];  // 616

  cv::Mat skip_map(h, 2 * w, CV_32F);  // 192, 1232
  cv::Mat map_xy(h, 2 * w, CV_32F);    // 192, 1232
  for (int r = 0; r < h; r++) {
    inFile.read((char *)(skip_map.ptr<float>(r)), 2 * w * skip_map.elemSize());
  }
  for (int r = 0; r < h; r++) {
    inFile.read((char *)(map_xy.ptr<float>(r)), 2 * w * map_xy.elemSize());
  }

  inFile.close();
  delete test;

  int fev_inp_w = input_width / 2;   // 1280 / 2 = 640
  int fev_inp_h = input_height / 2;  // 960 / 2 = 480

  if (one_step_remap_mode) {
    int roi_h = det_inp_h / roi_scale;      // 160 / 2.0 = 80
    int roi_w = det_inp_w / roi_scale;      // 352 / 2.0 = 176
    dh_shift /= roi_scale;                  // 30 / 2.0 = 15
    dw_shift /= roi_scale;                  // 0
    float dh = h - roi_h - dh_shift;        // 192 - 80 - 15  = 97
    float dw = (w - roi_w) / 2 - dw_shift;  // (616 - 176) / 2 = 220
    cv::Vec2f label;
    cv::Mat H_flip_xy = (cv::Mat_<float>(3, 3) << -1, 0, w, 0, -1, h, 0, 0, 1);
    cv::Mat map_roi_x(roi_h, roi_w, CV_32F);
    cv::Mat map_roi_y(roi_h, roi_w, CV_32F);
    for (int row = 0; row < roi_h; row++) {    // 80
      for (int col = 0; col < roi_w; col++) {  // 176
        matrix_mul_3x3(H_flip_xy, col + dw, row + dh, label,
                       multiply_float_model);
        int coord_y = label[1];
        int coord_x1 = label[0] * 2;
        int coord_x2 = label[0] * 2 + 1;
        float _col = map_xy.at<float>(coord_y, coord_x1) * fev_inp_w;
        float _row = map_xy.at<float>(coord_y, coord_x2) * fev_inp_h;
        map_roi_x.at<float>(row, col) = _col;
        map_roi_y.at<float>(row, col) = _row;
      }
    }
    map_roi_x *= 2;
    map_roi_y *= 2;
    cv::resize(map_roi_x, map_x, cv::Size(det_inp_w, det_inp_h),
               cv::INTER_LINEAR);
    cv::resize(map_roi_y, map_y, cv::Size(det_inp_w, det_inp_h),
               cv::INTER_LINEAR);
  } else {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        map_x.at<float>(i, j) = map_xy.at<float>(i, j * 2) * fev_inp_w;
        map_y.at<float>(i, j) = map_xy.at<float>(i, j * 2 + 1) * fev_inp_h;
      }
    }
  }

  return true;
}

void Util::binMatWrite(cv::Mat map_x, cv::Mat map_y, FILE *fpData) {
  fwrite(&map_x.cols, sizeof(int), 1, fpData);
  fwrite(&map_x.rows, sizeof(int), 1, fpData);

  std::vector<float> vec_texture(map_x.rows * map_x.cols * 2);

  for (int i = 0; i < map_x.rows; i++) {
    float *data_x = map_x.ptr<float>(i);
    float *data_y = map_y.ptr<float>(i);
    for (int j = 0; j < map_x.cols; j++) {
      vec_texture[(i * map_x.cols + j) * 2] = data_x[j];
      vec_texture[(i * map_x.cols + j) * 2 + 1] = data_y[j];
    }
  }
  fwrite(vec_texture.data(), vec_texture.size(), 4, fpData);
}

cv::Mat Util::normalize(cv::Mat srcimg, const std::vector<float> &mean_value,
                        const std::vector<float> &std_value,
                        const bool is_scale) {
  double e = 1.0;
  if (is_scale) {
    e /= 255.0;
  }
  srcimg.convertTo(srcimg, CV_32FC3, e);
  std::vector<cv::Mat> bgr_channels(3);
  cv::split(srcimg, bgr_channels);
  for (unsigned int i = 0; i < bgr_channels.size(); i++) {
    bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 / std_value[i],
                              (0.0 - mean_value[i]) / std_value[i]);
  }
  cv::Mat dstimg;
  cv::merge(bgr_channels, dstimg);
  return dstimg;
}

cv::Mat Util::det_wh_check_resize(const cv::Mat img, float &ratio_h,
                                  float &ratio_w) {
  cv::Mat resize_img;
  int w = img.cols;
  int h = img.rows;
  float ratio = 1.f;
  if (limit_type == "min") {
    int min_wh = std::min(h, w);
    if (min_wh < limit_side_len) {
      if (h < w) {
        ratio = float(limit_side_len) / float(h);
      } else {
        ratio = float(limit_side_len) / float(w);
      }
    }
  } else {
    int max_wh = std::max(h, w);
    if (max_wh > limit_side_len) {
      if (h > w) {
        ratio = float(limit_side_len) / float(h);
      } else {
        ratio = float(limit_side_len) / float(w);
      }
    }
  }

  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);

  resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
  resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);

  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  ratio_h = float(resize_h) / float(h);
  ratio_w = float(resize_w) / float(w);

  return resize_img;
}

cv::Mat Util::cls_resize(const cv::Mat &img,
                         const std::vector<int> &rec_image_shape) {
  cv::Mat resize_img;
  int imgH, imgW;
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  float ratio = float(img.cols) / float(img.rows);
  int resize_w;
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
  return resize_img;
}

cv::Mat Util::crnn_resize(const cv::Mat &img, float wh_ratio,
                          const std::vector<int> &rec_image_shape) {
  cv::Mat resize_img;
  int imgH, imgW;
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  imgW = int(imgH * wh_ratio);

  float ratio = float(img.cols) / float(img.rows);
  int resize_w;

  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
  cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                     int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                     {127, 127, 127});
  return resize_img;
}

}  // namespace ocr