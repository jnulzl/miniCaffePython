#encoding:utf-8

import os
import shutil

#��ͼ��ת����caffe���õ����ݸ�ʽ(lmdb��ʽ)
def ImagesToLMDB(resizeHeight,resizeWidth,imagePath,imageLabelsList,LMDBDir,flagGray = False,shuffle = False):
		'''
		resizeHeight:�ı���ͼ��߶ȣ����磺128��
		resizeWidth:�ı���ͼ���ȣ����磺256��
		
		imagesPath:���ͼ��ľ���·����windows�´�ͼ�������̷���ʼ�����磺
																	 D:\\MyImage\\					
																	 														
		imageLabelsList:���ͼ���ǩ�б�(һ��Ϊtxt)�ľ���·������ʽΪ��
																	imgName1.jpg(png) 0
																	imgName2.jpg(png) 0
																	imgName3.jpg(png) 1
																	....
																	imgNameN.jpg(png) m																										
																	
		LMDBDir:���ת���������ļ��еľ���·����
																	windows�´�ͼ�������̷���ʼ�����磺
																	D:\\MyImage\\trainLMDBImage,ת��������ݽ�������
																	trainLMDBImage�ļ�������
																	
		flagGray:�Ƿ��ԻҶ���ʽ��ͼ��Ĭ�ϣ�false��
		shuffle:�Ƿ����ͼƬ˳��Ĭ��:false��		
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

#����ͼ������(lmdb��ʽ)�ľ�ֵ
def computerImagesMean(LMDBDir,MeanFile):
		'''
		LMDBDir:��Ҫ����ľ�ֵ�����ļ���·��
		MeanFile:Ҫ����ľ�ֵ�ļ�·��,��չ��Ϊ��binaryproto	
		'''		
		temp = MeanFile + ".binaryproto"
		arg = LMDBDir + " " + temp		
		if (os.path.exists(temp)):
				os.remove(temp)
		
		#print "compute_image_mean " + arg
		os.system("compute_image_mean " + arg)
		
		return None		

#ѵ������		
def trainNet(solverFilePath):
	
		'''
		solverFilePath:ѵ���ļ�(.prototxt)		
		'''
		arg = solverFilePath
		os.system("caffe train -solver=" + arg)

		return None
		
#��������		
def testNet(netStructFile,weightsFile,batch=50,GPUID = False):
	
		'''
		netStructFile:����ṹ�ļ�
		weightsFile:Ȩֵ�����ļ�(.caffemodel)
		batch:����������Ĭ��Ϊ:50
		GPUID:ʹ�õ�GPU ID��,Ĭ��ΪFalse,��ʹ��GPU
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