#include <opencv2/imgproc/types_c.h>

#include <opencv2/opencv.hpp>

#include "ocr/inference.h"

int main() {
  ocr::InferencePtr ocr_handle = ocr::Inference::CreateInstance();

  std::string bin_dir_path = "../ocr/data/";
  std::string config_dir_path = "../ocr/data/";

  // initialize once
  int input_width = 1280;
  int input_height = 960;
  ocr_handle->initial(bin_dir_path, config_dir_path, input_width, input_height);

  // used for test
  cv::Mat test_img = cv::imread("../ocr/data/test_fev_1280x960.png");
  cv::cvtColor(test_img, test_img, CV_BGR2GRAY);
  unsigned char* yuv_addr = test_img.data;

  // online, call this
  ocr_handle->inference(yuv_addr);

  // acquire result
  auto res = ocr_handle->GetResult();
  for (auto& value : res) {
    std::cout << "text: " << value << std::endl;
  }
  // return this:  res[i].text  (digital)

  return 0;
}