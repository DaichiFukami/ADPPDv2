import listUp
import editImg

import numpy as np
import cupy as cp
xp = np


from learningPourDot import ADPPD
import chainer
import chainer.functions as F
import chainer.links as L

inPath = 'indata'
outPath = 'outdata'
charSize = 16*2

gpu = 0

#modelの設定
model = L.Classifier(ADPPD(),lossfun=F.mean_squared_error)#out-10種類(0-9の数字判別のため)
chainer.serializers.load_npz('mymodel.npz', model)

#GPU有無の判別
if gpu >= 0:
    chainer.cuda.get_device_from_id(gpu).use()
    model.to_gpu()
    xp = cp
    print('I use GPU and cupy')

#ファイルのリストアップ
listUp = listUp.ListUp()
fileList = listUp.inList(inPath)

#一時データ用ディレクトリの初期化
listUp.resetDir(outPath)

edit = editImg.EditImg(charSize,gpu)
print('進行状況:'+str(0)+'/'+str(len(fileList)))
for i in range(0, len(fileList)):
    inImgs = edit.preTeachImg(fileList[i][0])
    for y in range(len(inImgs)):
            for x in range(len(inImgs[0])):
                numpy = edit.img2numpy(inImgs[y][x])
                numpy2 = model.predictor(xp.array([numpy]).astype(xp.float32))
                numpy2 = F.relu(numpy2)
                numpy3 = numpy2.data[0]
                numpy3 = chainer.cuda.to_cpu(numpy3)
                inImgs[y][x] = edit.numpy2img(numpy3)
    outImg = edit.sutureImg(inImgs)
    outImg = outImg.convert("RGB")
    outImg.save(outPath+'/'+fileList[i][1]+'.png')
    print('進行状況:'+str(i+1)+'/'+str(len(fileList)))