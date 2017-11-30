import listUp
import editImg

from learningPourDot import ADPPD
from chainer import serializers

inPath = 'indata'
outPath = 'outdata'
charSize = 16*2

#modelの設定
model = ADPPD(4096,3072)
serializers.load_npz('mymodel.npz', model)

#ファイルのリストアップ
listUp = listUp.ListUp()
fileList = listUp.inList(inPath)

#一時データ用ディレクトリの初期化
listUp.resetDir(outPath)

edit = editImg.EditImg(charSize)
print('進行状況:'+str(0)+'/'+str(len(fileList)))
for i in range(0, len(fileList)):
    inImgs = edit.preTeachImg(fileList[i][0])
    for y in range(len(inImgs)):
            for x in range(len(inImgs[0])):
                numpy = edit.img2numpy(inImgs[y][x])
                #numpy2 = model(numpy)
                inImgs[y][x] = edit.numpy2img(numpy)
    outImg = edit.sutureImg(inImgs)
    outImg = outImg.convert("RGB")
    outImg.save(outPath+'/'+fileList[i][1]+'.png')
    print('進行状況:'+str(i+1)+'/'+str(len(fileList)))
