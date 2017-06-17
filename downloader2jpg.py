import numpy as np
import os
import cv2
from struct import unpack
from scipy import misc
from urllib import request
import gzip

url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url_train_label = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
url_test_label = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

urls = {
        'train': [url_train_image, url_train_label],
        'test': [url_test_image, url_test_label]
        }
# 파일 읽기
# read files
for stage in ["train", "test"]:
    # get files from internet
    fp_image = gzip.open(request.urlretrieve(urls[stage][0])[0], 'rb')
    fp_label = gzip.open(request.urlretrieve(urls[stage][1])[0], 'rb')
    # make directories
    cwd = os.getcwd()
    wd = "%s/output" % (cwd)
    if not os.path.exists(wd):
        os.mkdir(wd)
    wd += '/%s' % (stage)
    if not os.path.exists(wd):
        os.mkdir(wd)
    for i in range(10):
        if not os.path.exists("%s/%d" % (wd, i)):
            os.mkdir("%s/%d" % (wd, i))
    # 사용할 변수 초기화
    img = np.zeros((28, 28))  # 이미지가 저장될 부분

    index = 0

    # drop header info?
    s = fp_image.read(16)    # read first 16byte
    l = fp_label.read(8)     # read first  8byte

    # 숫자 데이터를 읽어서 해당하는 데이터를 지정하고 출력
    k = 0                    # 테스트용 index
    # read mnist and show number
    while True:
        # 784바이트씩 읽음
        s = fp_image.read(784)
        # 1바이트씩 읽음
        l = fp_label.read(1)

        if not s:
            break
        if not l:
            break
        index = int(l[0])
        # unpack
        img = np.reshape(unpack(len(s) * 'B', s), (28, 28))
        # resize from 28x28 to 10x10
        resized = misc.imresize(img, (10, 10))
        # binarization
        ret, thresh = cv2.threshold(resized, 75, 255, cv2.THRESH_BINARY)
        # save binary image
        filename = '%s/%s/%s' % (wd, index, k)
        misc.imsave(filename+'_rsz.jpg', resized)
        misc.imsave(filename+'_bin.jpg', thresh)
        misc.imsave(filename+'_org.jpg', img)
        k += 1
    print("read done", k, stage)
