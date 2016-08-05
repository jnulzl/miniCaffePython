#encoding:utf-8

import os
import shutil

#将图像转换成caffe可用的数据格式(lmdb格式)
def ImagesToLMDB(resizeHeight,resizeWidth,imagePath,imageLabelsList,LMDBDir,flagGray = False,shuffle = False):
		'''
		resizeHeight:改变后的图像高度，例如：128；
		resizeWidth:改变后的图像宽度，例如：256；
		
		imagesPath:存放图像的绝对路径。windows下从图像所在盘符开始，例如：
																	 D:\\MyImage\\					
																	 														
		imageLabelsList:存放图像标签列表(一般为txt)的绝对路径。格式为：
																	imgName1.jpg(png) 0
																	imgName2.jpg(png) 0
																	imgName3.jpg(png) 1
																	....
																	imgNameN.jpg(png) m																										
																	
		LMDBDir:存放转换后数据文件夹的绝对路径；
																	windows下从图像所在盘符开始，例如：
																	D:\\MyImage\\trainLMDBImage,转换后的数据将保存在
																	trainLMDBImage文件夹下面
																	
		flagGray:是否以灰度形式打开图像，默认：false；
		shuffle:是否打乱图片顺序，默认:false。		
		'''
		strImageSize = " -resize_height=" + str(resizeHeight) + " -resize_width=" + str(resizeWidth) + " "
		temp = strImageSize + imagePath + " " + imageLabelsList + " " + LMDBDir + " "
		arg = temp
		if (flagGray):
				arg = " -gray" + temp
		if (shuffle):
				arg = " -shuffle" + temp
		if (flagGray and shuffle):
				arg = " -gray" + " -shuffle" + temp
		
		if (os.path.exists(LMDBDir)):
				shutil.rmtree(LMDBDir)
		os.system("convert_imageset " + arg)
		
		return None

#计算图像数据(lmdb格式)的均值
def computerImagesMean(LMDBDir,MeanFile):
		'''
		LMDBDir:需要计算的均值数据文件夹路径
		MeanFile:要保存的均值文件路径,扩展名为：binaryproto	
		'''		
		temp = MeanFile + ".binaryproto"
		arg = LMDBDir + " " + temp		
		if (os.path.exists(temp)):
				os.remove(temp)
		
		#print "compute_image_mean " + arg
		os.system("compute_image_mean " + arg)
		
		return None		

#训练网络		
def trainNet(solverFilePath):
	
		'''
		solverFilePath:训练文件(.prototxt)		
		'''
		arg = solverFilePath
		os.system("caffe train -solver=" + arg)

		return None
		
#测试网络		
def testNet(netStructFile,weightsFile,batch=50,GPUID = False):
	
		'''
		netStructFile:网络结构文件
		weightsFile:权值参数文件(.caffemodel)
		batch:迭代次数，默认为:50
		GPUID:使用的GPU ID号,默认为False,不使用GPU
		'''
		temp = netStructFile + " -weights=" + weightsFile
		arg = temp + " -iterations=" + str(batch)
		if (0 is GPUID):
				arg = temp + " -gpu " + str(GPUID)
			
		if(GPUID):
				arg = temp + " -gpu " + str(GPUID)
		
		#print "caffe test -model=" + arg
		#print type("caffe test -model=" + arg)
		os.system("caffe test -model=" + arg)

		return None
						
'''
resizeHeight = 32
resizeWidth = 32

imagePath = "../data/img/"
imageLabelsList = "../train.txt"
LMDBDir = 	"./train_lmdb"
	
ImagesToLMDB(resizeHeight,resizeWidth,imagePath,imageLabelsList,LMDBDir,False,True)

solverFilePath = "./lenet_solver.prototxt"

trainNet(solverFilePath)
'''