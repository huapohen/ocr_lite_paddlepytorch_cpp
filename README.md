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


有问题留言，主要是跑通后，就跑去做神经网络端到端AVM了。  
（If you have any questions, leave a Issues, mainly after running through, then go to do neural network end-to-end AVM.）  

上传之前，代码的3方库抽出去了，可能有些没改对，编译或跑不起来，先看看能不能自行解决吧，留个issue我不一定能及时看到  
(Before uploading, the 3-party library of the code was pulled out. Some of them may not be corrected, and they cannot be compiled or run. Let’s see if we can solve it by ourselves. I may not be able to see it in time if you leave an issue.)


