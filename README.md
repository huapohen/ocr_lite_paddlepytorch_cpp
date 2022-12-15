git submodule update --init  
若首次clone，从git lfs 上拉数据（If clone for the first time, pull data from git lfs）    
git lfs install  
git lfs pull  
mkdir build && cd build  
cmake ..  
make -j8  


上面拉子模块不通，直接下，再放到工程目录里（The above pull sub-module fails, just download it and put it in the project directory）：  
3rdparty/jsoncpp/  
3rdparty/ncnn/  
3rdparty/opencv_3.4/  
3rdparty/opencv_4.3.0/  
  
  
git@github.com:huapohen/ncnn.git  
git@github.com:huapohen/opencv_3.4.git  
git@github.com:huapohen/opencv_4.3.0.git  


1> build
2> run this:  cd build && ./build/ocr  

ocr/inference_impl.cc的std::vector<std::string> InferenceImpl::GetResult() 这个函数里，把筛选条件去掉（目前是只出字母和数字），就是所有的字符了（In this function, remove the filter conditions (currently only letters and numbers are displayed), that is, to detect all characters）  

字符文件路径（character file path）： ocr/data/keys.txt   5532 characters


这是鱼眼图像的输入，还要切换模式为正常图像输入，后续再改  
This is the input of the fisheye image, and the mode needs to be switched to normal image input, which will be changed later
