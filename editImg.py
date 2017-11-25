from PIL import Image
import math
import os

class EditImg:
    charSize = 0
    #色相分の水増し数(1,2,3,6,12)
    hs = 6
    #背景色(黒)
    back =(0,0,0)
    def __init__(self,charSize):
        self.charSize = charSize
    def expCanvas(self,inImg):
        xChar = math.ceil(inImg.size[0]/self.charSize)
        yChar = math.ceil(inImg.size[1]/self.charSize)

        xFix = xChar*self.charSize
        yFix = yChar*self.charSize

        size = (xFix,yFix)

        xSt = math.floor((xFix-inImg.size[0])/2)
        ySt = math.floor((yFix-inImg.size[1])/2)

        start = (xSt,ySt)

        outImg = Image.new('HSV',size,self.back)
        outImg.paste(inImg, start)

        return outImg
    def writeImg(self,inImg,outDel,fileName,fileX,fileY):
        if(os.path.isdir(outDel)==False):
            os.mkdir(outDel)
        outImgs =[[0 for i in range(8)] for j in range(self.hs)]
        outImgs[0][0] = inImg
        outImgs[0][4] = outImgs[0][0].transpose(Image.ROTATE_90)
        for i in range(0, 8, 4):
            outImgs[0][i+1] = outImgs[0][i].transpose(Image.FLIP_LEFT_RIGHT)
            outImgs[0][i+2] = outImgs[0][i].transpose(Image.FLIP_TOP_BOTTOM)
            outImgs[0][i+3] = outImgs[0][i+1].transpose(Image.FLIP_TOP_BOTTOM)
        ctr = 0
        files = os.listdir(outDel)
        for i in files:
            ctr = ctr + 1
        for j in range(1,self.hs):
            for i in range(0, 8):
                h, s, v = outImgs[0][i].split()
                j2 = (255/self.hs)*j
                h2 = h.point(lambda h: round((h+j2)%255,0))
                outImgs[j][i] = Image.merge("HSV", (h2 , s, v))
                outImg = outImgs[j][i].convert("RGB")
                outImg.save(outDel+'/'+str(ctr)+'.png')
                ctr = ctr + 1
                #outImg.save(outDel+'/'+fileName+'_'+str(i)+'_'+str(j)+'_'+str(fileY).zfill(3)+'_'+str(fileX).zfill(3)+'.png')
    def quarryImg(self,inImg):
        xCh = int(inImg.size[0]/self.charSize)*2-1
        yCh = int(inImg.size[1]/self.charSize)*2-1
        outImgs = [[0 for i in range(xCh)] for j in range(yCh)]
        for j in range(0, yCh):
                for i in range(0, xCh):
                    xSt = i*int(self.charSize/2)
                    ySt = j*int(self.charSize/2)
                    outImgs[j][i] = inImg.crop((xSt, ySt, (xSt+self.charSize), (ySt+self.charSize)))
        return outImgs

    def outImg(self,outDel,path):
        img = Image.open(path).convert("HSV")
        img = self.expCanvas(img)
        imgsqua = self.quarryImg(img)
        for y in range(0, len(imgsqua)):
                for x in range(0, len(imgsqua[0])):
                    fileNames = os.path.basename(path)
                    fileNames = os.path.splitext(fileNames)
                    fileName = fileNames[0][:-2]
                    self.writeImg(imgsqua[y][x],outDel,fileName,x,y)

"""
edit = EditImg()
edit.outImg('que','traindata/0000_i.bmp')
print('書き出し完了')
"""