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


目前只输出 数字和字母，如果想输出所有5532个字符，改代码中条件，即可 ocr/inference_impl.cc---->GetResult()--->211-212行  
in  ocr/inference_impl.cc----> std::vector<std::string> InferenceImpl::GetResult() ----> filter out digit 0~9 and a~z(A~Z)    
if you want output all 5532 characters, you can change the judgment condition 211-212 lines:  "if (true) { // only output 0~9 and a~z(A~Z)  ---->  if (false) {"   

目前只输出 ocr的一个结果，改ocr/inference_impl.cc---->GetResult()--->238行，最直接   
the num of ocr output results can be change in ocr/inference_impl.cc---->GetResult()---> 238 line, change the judgment condition  


输入图像目前可以是任意图，把 ocr/data/ocr_config.json里的non_bev_test_mode，置为true 即可。因为图输入后，支持去畸变，不走去畸变的通道即可，就是任意图输入，直接做检测识别。  
目前把图进行了一些前处理，可在代码中去掉  
(The input image can be any image at present, just set non_bev_test_mode in ocr/data/ocr_config.json to 'true'.)   
(At present, some pre-processing has been performed on the graph, which can be removed in the code)  

