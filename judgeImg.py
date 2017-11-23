import os
from PIL import Image

class JudgeImg:
    def __init__(self):
        pass
    #ファイル名の有無の確認
    def judgeImgName(self,path):
        bmp = path+'.bmp'
        png = path+'.png'
        pcx = path+'.pcx'
        if(os.path.exists(bmp) == True):
            return bmp
        elif(os.path.exists(png) == True):
            return png
        elif(os.path.exists(pcx) == True):
            return pcx
        else:
            return 'null'
    #que,ansのサイズの一致確認
    def judgeImgSize(self,path):
        queName = (path+'_i')
        ansName = (path+'_o')
        quePath = self.judgeImgName(queName)
        ansPath = self.judgeImgName(ansName)

        if(quePath != 'null' and ansPath != 'null' ):
            queImg = Image.open(quePath)
            ansImg = Image.open(ansPath)
        else:
            return ('null','null')

        if (queImg.size[0] == ansImg.size[0] and queImg.size[1] == ansImg.size[1]):
            return (quePath,ansPath)
        else:
            return ('null','null')
