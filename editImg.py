import numpy as np
import cupy as cp
xp = np

from PIL import Image
import math
import os

class EditImg:
    charSize = 0
    #色相分の水増し数(1,2,3,6,12)
    hs = 3
    #背景色(黒)
    back =(0,0,0)
    def __init__(self,charSize,gpu):
        self.charSize = charSize
        if(gpu>=0):
            xp = cp

    def expCanvas(self,inImg):
        xChar = math.ceil(inImg.size[0]/self.charSize)
        yChar = math.ceil(inImg.size[1]/self.charSize)

        xFix = xChar*self.charSize
        yFix = yChar*self.charSize

        xSt = math.floor((xFix-inImg.size[0])/2)
        ySt = math.floor((yFix-inImg.size[1])/2)

        outImg = Image.new('HSV',(xFix,yFix),self.back)
        outImg.paste(inImg,(xSt,ySt))

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
        for j in range(0,self.hs):
            for i in range(0, 8):
                h, s, v = outImgs[0][i].split()
                j2 = (255/self.hs)*j
                h2 = h.point(lambda h: round((h+j2)%255,0))
                outImgs[j][i] = Image.merge("HSV", (h2 , s, v))
                outImg = outImgs[j][i].convert("RGB")
                outImg.save(outDel+'/'+str(ctr)+'.png')
                ctr = ctr + 1

    def quarryImg(self,inImg,moveLength):
        self.moveLength = int(moveLength)
        xCh = int((inImg.size[0]-self.charSize)/self.moveLength+1)
        yCh = int((inImg.size[1]-self.charSize)/self.moveLength+1)
        outImgs = [[0 for i in range(xCh)] for j in range(yCh)]
        for j in range(0, yCh):
                for i in range(0, xCh):
                    xSt = i*self.moveLength
                    ySt = j*self.moveLength
                    outImgs[j][i] = inImg.crop((xSt, ySt, (xSt+self.charSize), (ySt+self.charSize)))
        return outImgs

    def sutureImg(self,inImgs):
        size = inImgs[0][0].size[0]
        xFix = size*len(inImgs[0])
        yFix = size*len(inImgs)
        outImg = Image.new('HSV',(xFix,yFix),self.back)
        for y in range(len(inImgs)):
            for x in range(len(inImgs[0])):
                outImg.paste(inImgs[y][x],(x*size,y*size))
        return outImg

    def img2numpy(self,img):
        imgF = xp.asarray(img, dtype=xp.float32)
        numpy = []
        for z in range(0,3):
            for y in range(0,self.charSize):
                for x in range(0,self.charSize):
                        numpy.append(xp.float32(imgF[y][x][z])/255.0)
        numpy = xp.asarray(numpy,dtype=xp.float32)
        return numpy

    def numpy2img(self,numpy):
        size = round(math.sqrt(len(numpy)/3))
        outImg = Image.new('HSV',(size,size),self.back)
        for y in range(0,self.charSize):
            for x in range(0,self.charSize):
                outImg.putpixel((x, y),(
                    int(round(numpy[x+y*size]*255)),
                    int(round(numpy[x+y*size+size*size]*255)),
                    int(round(numpy[x+y*size+size*size*2]*255))
                    ))
        return outImg

    def outImg(self,outDel,path):
        img = Image.open(path).convert("HSV")
        img = self.expCanvas(img)
        imgsQua = self.quarryImg(img,int(self.charSize/2))
        for y in range(0, len(imgsQua)):
                for x in range(0, len(imgsQua[0])):
                    fileNames = os.path.basename(path)
                    fileNames = os.path.splitext(fileNames)
                    fileName = fileNames[0][:-2]
                    self.writeImg(imgsQua[y][x],outDel,fileName,x,y)

    def preTeachImg(self,path):
        img = Image.open(path).convert("HSV")
        img = self.expCanvas(img)
        imgsQua = self.quarryImg(img,int(self.charSize))
        return imgsQua



"""
edit = EditImg(32)
img = Image.open('alldata/0000_i.bmp').convert("HSV")
img = edit.expCanvas(img)
imgsqua = edit.quarryImg(img,32)
imgOut = edit.sutureImg(imgsqua)
imgOut = imgOut.convert("RGB")
imgOut.save('alldata/sample_out.png')
"""


"""
edit = EditImg(32)
inImg = Image.open('alldata/sample.png').convert("HSV")
numpy = edit.img2numpy(inImg)
image = edit.numpy2img(numpy)
image = image.convert("RGB")
image.save('alldata/sample_out.png')
print('書き出し完了')
"""