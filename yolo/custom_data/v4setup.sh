
git clone https://github.com/AlexeyAB/darknet.git ../darknetv4

cp ./yolo-custom.data ../darknetv4/build/darknet/x64/data/obj.data
cp ./custom.names  ../darknetv4/build/darknet/x64/data/obj.names
cp ./cfg/yolov4-custom.cfg  ../darknetv4/yolo-obj.cfg
cp ./train.txt  ../darknetv4/build/darknet/x64/data/
cp ./test.txt  ../darknetv4/build/darknet/x64/data/

mkdir  ../darknetv4/build/darknet/x64/data/obj/
cp -r ./images/*.jpg  ../darknetv4/build/darknet/x64/data/obj/  
cp -r ./labels/*.txt ../darknetv4/build/darknet/x64/data/obj/

cd ../darknetv4/
#wget "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
sed -i.bu 's/OPENCV=0/OPENCV=1/' Makefile
#sed -i.bu 's/CUDNN=0/CUDNN=1/' Makefile 
#sed -i.bu 's/GPU=0/GPU=1/' Makefile 
#sed -i.bu 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile 
 
#make
#./darknet detector train build/darknet/x64/data/obj.data yolo-obj.cfg yolov4.conv.137 -dont_show
