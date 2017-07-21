import numpy as np
import chainer
import argparse

import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer import Variable
from chainer import serializers

from fix_data_class import OneDimSensorClass

"""
    NeuralAE:
        lstmを利用して異常値を検知するのではなく,
        時系列データを50次元で, 通常時のデータのみを学習器にいれたモデル
"""


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


# gpu
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID')
args = parser.parse_args()


# raw data class
OneDimSensor = OneDimSensorClass()
OneDimSensor.load_csv()
train = np.array(OneDimSensor.train_ary0, dtype=np.float32)
test = OneDimSensor.test_ary0

# model setup
model = NeuralAE()
optimizer = optimizers.Adam()
optimizer.setup(model)

# for cuda
xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# training
n = len(train)
bs = int(len(train)/10)
loss_length = len(test)
test_loss_ary = []

for i in range(300):
    sffidx = np.random.permutation(n)
    for j in range(0, n, bs):
        x = train[sffidx[i:(i+bs) if (i+bs) < n else n]]
        model.zerograds()
        loss = model(x)
        loss.backward()
        optimizer.update()

    # テストデータの評価
    total_loss = 0
    for data in test:
        y = Variable(data, volatile=True).reshape(1, 50)
        # import ipdb; ipdb.set_trace()
        loss = model.forward(y)
        total_loss += np.mean(np.abs(loss.data))
    print('{} epoch, test loss'.format(i), total_loss/loss_length)
    test_loss_ary.append(total_loss/loss_length)
    serializers.save_npz('model/NeuralAE_{}'.format(i), model)




