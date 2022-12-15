git submodule update --init  
若首次clone，从git lfs 上拉数据  
git lfs install  
git lfs pull  
mkdir build && cd build  
cmake ..  
make -j8  


上面拉子模块不通，直接下，再放到工程目录里：  
3rdparty/jsoncpp/  
3rdparty/ncnn/  
3rdparty/opencv_3.4/  
3rdparty/opencv_4.3.0/  
  
  
git@github.com:huapohen/ncnn.git  
git@github.com:huapohen/opencv_3.4.git  
git@github.com:huapohen/opencv_4.3.0.git  


1> build
2> run this:  cd build && ./build/ocr  

ocr/inference_impl.cc的std::vector<std::string> InferenceImpl::GetResult() 这个函数里，把筛选条件去掉（目前是只出字母和数字），就是所有的字符了  

字符文件路径： ocr/data/keys.txt  5532个
