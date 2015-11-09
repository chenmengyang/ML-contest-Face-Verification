# encoding=utf-8
__author__ = 'cmy'
import os
from libsvm.svmutil import *

os.chdir('/Users/cmy/Documents/Machine_Learning/contest/5.SVM/Python/test_data')

y, x = svm_read_problem('train1.txt')#读入训练数据
yt, xt = svm_read_problem('test1.txt')#训练测试数据
m = svm_train(y, x )#训练
svm_predict(yt,xt,m)#测试