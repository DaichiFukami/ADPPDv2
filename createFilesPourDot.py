import listUp
import editImg

tdPath = 'alldata'
quePath ='que'
ansPath = 'ans'
trainQuePath ='trainQue'
trainAnsPath = 'trainAns'
testQuePath ='testQue'
testAnsPath = 'testAns'
charSize = 16*2

#ファイルのリストアップ
listUp = listUp.ListUp()
fileList = listUp.tdList(tdPath)
#一時データ用ディレクトリの初期化
listUp.resetDir(quePath)
listUp.resetDir(ansPath)

edit = editImg.EditImg(charSize)
print('進行状況:'+str(0)+'/'+str(len(fileList)))
for i in range(0, len(fileList)):
    edit.outImg(quePath,fileList[i][0])
    edit.outImg(ansPath,fileList[i][1])
    print('進行状況:'+str(i+1)+'/'+str(len(fileList)))
listUp.splitDeta(quePath,ansPath,trainQuePath,trainAnsPath,
                 testQuePath,testAnsPath,0.2)