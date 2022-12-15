git submodule update --init  
若首次clone，从git lfs 上拉数据  
git lfs install  
git lfs pull  
mkdir build && cd build  
cmake ..  
make -j8  


上面拉子模块不通，直接下，再放到工程目录里：
3rdparty/
   jsoncpp/
   ncnn/
   opencv_3.4/
   opencv_4.3.0/
