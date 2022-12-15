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


编译后  
执行：cd build && ./build/ocr  

