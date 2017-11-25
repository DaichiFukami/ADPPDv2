import listUp
import editImg
import os
import shutil

tdPath = 'traindata'
quePath ='que'
ansPath = 'ans'

def resetDir(path):
    if(os.path.isdir(path)==True):
        shutil.rmtree(path)
    os.mkdir(path)

#ファイルのリストアップ
listUp = listUp.ListUp(tdPath)
fileList = listUp()
listUp = None
#一時データ用ディレクトリの初期化
resetDir(quePath)
resetDir(ansPath)

edit = editImg.EditImg()
for i in range(0, len(fileList)):
    edit.outImg(quePath,fileList[i][0])
    edit.outImg(ansPath,fileList[i][1])
    print('進行状況:'+str(i+1)+'/'+str(len(fileList)))