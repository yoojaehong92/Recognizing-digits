import numpy as np
import sys
import os
from array import array

from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from scipy import misc

#파일 읽기 
fp_image = open('train-images-idx3-ubyte','rb')
fp_label = open('train-labels-idx1-ubyte','rb')

#사용할 변수 초기화
img = np.zeros((28,28)) #이미지가 저장될 부분
lbl = [ [],[],[],[],[],[],[],[],[],[] ] #숫자별로 저장 (0 ~ 9)
d = 0
l = 0
index=0 

s = fp_image.read(16)    #read first 16byte
l = fp_label.read(8)     #read first  8byte


#숫자 데이터를 읽어서 해당하는 데이터를 지정하고 출력 
k=0 #테스트용 index
#read mnist and show number
while True:    
    s = fp_image.read(784) #784바이트씩 읽음
    l = fp_label.read(1) #1바이트씩 읽음

    if not s:
        break; 
    if not l:
        break;

    index = int(l[0]) 
    #print(k,":",index) 

#unpack
    img = np.reshape( unpack(len(s)*'B',s), (28,28)) 
    resized = misc.imresize(img,(10,10))
    ret,thresh = cv2.threshold(resized,75,255,cv2.THRESH_BINARY)
    #lbl[index].append(ret) #각 숫자영역별로 해당이미지를 추가
    k=k+1
    filename = '%s/%s.jpg'%(index,k)
    misc.imsave(filename, thresh)
    #plt.subplot(1,2,1),plt.imshow(resized,cmap = cm.binary)
    #plt.subplot(1,2,2),plt.imshow(thresh,cmap = cm.binary)
    #plt.show()
#print(img)
print(img.shape,img.dtype)
#plt.imshow(img,cmap = cm.binary) #binary형태의 이미지 설정
#plt.show()
#print(np.shape(lbl)) #label별로 잘 지정됬는지 확인

print("read done",k)

# lbl 에 이미지들 있음.

#resized = misc.imresize(img,(10,10))


#plt.title(index)
#plt.show()
