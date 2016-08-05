	本脚本用Python将C++版的Caffe进行了封装，仅仅是个人爱好而已。
利用Caffe进行图像分类的一般步骤：
		1、生成自己的图像标签,一般用txt保存起来，格式如下：
									图像名1.jpg 0
									图像名2.jpg 0
									...
									图像名M.jpg 1
									图像名(M+1).jpg 1
									...
									图像名N.jpg k
									图像名(N+1).jpg k
		2、将图像数据转化为Caffe可用的格式，一般为lmdb;
		3、计算所有图像像素数据的均值;
		4、编写配置文件：xx-Solver-xx.prototxt;
		5、编写网络结构文件：xx-train_test-xx.prototxt;
		6、训练网络;
		7、对网络进行测试。
		

									
							