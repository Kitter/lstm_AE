import numpy as np
import csv
import random

"""
    0とされもののうち, 90%を訓練データにする, 10%をバリデーションデータ
    そのうち80%を訓練データにし, 20%をテストデータにする
    推測するときは圧倒的に多い1と, 10%のバリデーションデータが分類できているかをプロットしてみる
    OneDimSensor: 一次元のセンサーを5秒ごとにくぎってデータ化したもの
"""


class OneDimSensorClass(object):
    def __init__(self):
        self.train_ary0 = []
        self.test_ary0 = []
        self.validation_ary0 = []
        self.validation_ary1 = []

    def load_csv(self):
        with open('confused_data/Raw/0.csv', 'r') as f:
            ary0 = [np.array(row[0:50], dtype=np.float32) for row in csv.reader(f)]

        with open('confused_data/Raw/1.csv', 'r') as f:
            self.validation_ary1 = [np.array(row[0:50], dtype=np.float32) for row in csv.reader(f)]

        random.shuffle(ary0)
        self.train_ary0 = ary0[:int(len(ary0)/10*9)]
        self.validation_ary0 = ary0[int(len(ary0)/10*9):]

        random.shuffle(self.train_ary0)
        copy_ary0 = self.train_ary0
        self.train_ary0 = copy_ary0[:int(len(ary0)/5*4)]
        self.test_ary0 = copy_ary0[int(len(ary0)/5*4):]

        print('train_0 {}, test_0 {}, '
              'validation_0 {}, validation_1 {}'.format(len(self.train_ary0), len(self.test_ary0),
                                                        len(self.validation_ary0), len(self.validation_ary1)))
