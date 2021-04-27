
git clone https://github.com/pjreddie/darknet.git ../darknetv3

cd ../darknetv3/
sed -i.bu 's/OPENCV=0/OPENCV=1/' Makefile
sed -i.bu 's/CUDNN=0/CUDNN=1/' Makefile 
sed -i.bu 's/GPU=0/GPU=1/' Makefile 
 
make
./darknet detector train ../custom_data/yolo-custom.data ../custom_data/yolov3-custom.cfg ../custom_data/darknet53.conv.74
