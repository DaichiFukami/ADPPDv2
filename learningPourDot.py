import numpy as np
import cupy as cp
xp = np

from PIL import Image
import glob
import sys

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

trainQuePath ='trainQue'
trainAnsPath = 'trainAns'
testQuePath ='testQue'
testAnsPath = 'testAns'
charSize = 16*2
outUnit = 3*charSize*charSize
unitSize = round(outUnit*4/3)

class DatasetPourDot(chainer.dataset.DatasetMixin):
    def __init__(self,quePath,ansPath):
        self.quePath = quePath
        self.ansPath = ansPath
        #データの数を吐く
        n1 = len(glob.glob(quePath + '/*.png'))
        n2 = len(glob.glob(ansPath + '/*.png'))
        if n1 != n2:
            print ('エラー:学習用データが不正です')
            sys.exit(1)
        self._size = n1
        return

    def __len__(self):
        #ここで入っているデータの一覧を入れる
        return self._size

    def get_example(self, idx):
        if idx < 0 or idx >= self._size:
            print ('エラー:不正なインデックス')
            sys.exit(1)
        #ここから問題データ読み込み
        queImg = Image.open(self.quePath + ('/')+str(idx)+('.png' )).convert("HSV")
        ansImg = Image.open(self.ansPath + ('/')+str(idx)+('.png' )).convert("HSV")
        queNum = self.img2numpy(queImg)
        ansNum = self.img2numpy(ansImg)
        return queNum,ansNum

    def img2numpy(self,Img):
        imgF = xp.asarray(Img, dtype=xp.float32)
        numpy = []
        for z in range(0,3):
            for y in range(0,charSize):
                for x in range(0,charSize):
                        numpy.append(xp.float32(imgF[y][x][z])/255.0)
        numpy = xp.asarray(numpy,dtype=xp.float32)
        return numpy

class ADPPD(chainer.Chain):
    def __init__(self):
        n_units = unitSize
        n_out = outUnit
        super(ADPPD, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))#入力をl1で変換、さらに活性化関数で変換
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class ADPPD_CNN(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(ADPPD_CNN, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))#入力をl1で変換、さらに活性化関数で変換
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
def main():
    #初期設定
    parser = argparse.ArgumentParser(description='Chainer')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='バッチサイズ')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='エポック数')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPUの有無')
    parser.add_argument('--out', '-o', default='result',
                        help='リサルトファイルのフォルダ')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=unitSize,
                        help='中間層の数')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    ##MLPをここで引っ張る
    model = L.Classifier(ADPPD(args.unit, outUnit),lossfun=F.mean_squared_error)#out-10種類(0-9の数字判別のため)
    model.compute_accuracy = False
    #GPU有無の判別
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cp
        print('I use GPU and cupy')

    ##optimizerのセット
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    ##データセットをchainerにセット
    train_data = DatasetPourDot(trainQuePath,trainAnsPath)
    test_data = DatasetPourDot(testQuePath,testAnsPath)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)

    ##updater=重みの調整、今回はStandardUpdaterを使用
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    ##updaterをtrainerにセット
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    ##評価の際、Evaluatorを使用
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))


    ##途中経過の表示用の記述
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    #中断データの有無、あれば続きから
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    #実験開始、trainerにお任せ
    trainer.run()
    #CPUで計算できるようにしておく
    model.to_cpu()
    #npz形式で書き出し
    chainer.serializers.save_npz(args.out+'/mymodel.npz', model)


if __name__ == '__main__':
    main()