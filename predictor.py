import numpy as np
import chainer
import argparse

import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import Variable
from chainer import serializers
import matplotlib.pyplot as plt

from fix_data_class import OneDimSensorClass


class NeuralAE(chainer.Chain):
    # TODO: 中間層のノードの数は考察必須
    def __init__(self):
        super().__init__(
            l1=L.Linear(50, 2),
            l2=L.Linear(2, 50),
        )

    def __call__(self, x):
        h = self.forward(x)
        return F.mean_squared_error(h, x)

    def forward(self, x):
        h = F.sigmoid(self.l1(x))
        # import ipdb; ipdb.set_trace()
        h = self.l2(h)
        return h

    def AutoEncoder(self, x):
        h = F.sigmoid(self.l1(x))
        return h

OneDimSensor = OneDimSensorClass()
OneDimSensor.load_csv()
validation0 = OneDimSensor.validation_ary0
validation0 = [(row, 0) for row in validation0]
validation1 = OneDimSensor.validation_ary1
validation1 = [(row, 1) for row in validation1]
validations = validation0 + validation1

# setup model
model = NeuralAE()
chainer.serializers.load_npz('model/NeuralAE_299', model)
x_plt_ary0 = []
y_plt_ary0 = []
x_plt_ary1 = []
y_plt_ary1 = []

for validation in validations:
    data = model.AutoEncoder(validation[0].reshape(1, 50)).data
    #import ipdb; ipdb.set_trace()
    if validation[1] == 0:
        x_plt_ary0.append(data[0][0])
        y_plt_ary0.append(data[0][1])
    else:
        x_plt_ary1.append(data[0][0])
        y_plt_ary1.append(data[0][1])

x_plt_ary0 = np.array(x_plt_ary0, dtype=np.float32)
y_plt_ary0 = np.array(y_plt_ary0, dtype=np.float32)
x_plt_ary1 = np.array(x_plt_ary1, dtype=np.float32)
y_plt_ary1 = np.array(y_plt_ary1, dtype=np.float32)
plt.scatter(x_plt_ary0, y_plt_ary0, c='b')
plt.scatter(x_plt_ary1, y_plt_ary1, c='r', alpha=0.1)
plt.show()