# Recognizing-digits
2017_Spring_AI_TermProject

## Enviroment
Python3.6.1
numpy (1.12.1)
Pillow (4.1.1)
scipy (0.19.0)

## Usage
$python3 main.py train
```
10 ~ 40개의 hidden nodes를 바꿔가면서 트레이닝한다.
```
$python3 main.py test
```
트레이닝된 네트워크들에 test set을 이용하여 테스트한 후
가장 정확도가 높은 네트워크를 출력한다.
```
$python3 downloader2jpg.py
```
MNIST 데이터 셋을 다운로드받아 학습에 필요한 데이터인
원본 이미지, 10x10 이미지, 10x10 이진화 이미지로 변환하여 저장한다.
```
$python3 gui.py
```
학습된 네트워크를 이용하여 프로그램을 통해 숫자를 그려보고
그에 해당하는 예측값을 출력해주는 프로그램
```



## Goal
Classify a handwritting number ( 0 ~ 9 ) of 10 x 10 pixel image

## DataSet
resized MNIST data from 28x28 to 10x10 and binarization 

## Neural Network Structure
input layer     - 100 nodes
1 hidden layer  - 10 ~ 31 nodes
output layer - 10 nodes

## Reference

http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python
