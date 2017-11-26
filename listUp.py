import fnmatch
import os
from os.path import join, relpath
from glob import glob
import random
import shutil

import judgeImg

class ListUp:
    def __init__(self):
        pass
    def resetDir(self,path):
        if(os.path.isdir(path)==True):
            shutil.rmtree(path)
        os.mkdir(path)
    def tdList(self,tdPath):
        self.tdPath = tdPath
        allList = [relpath(x, self.tdPath) for x in glob(join(self.tdPath, '*'))]
        inList = []
        outList = []
        ji = judgeImg.JudgeImg()
        #拡張子を削除
        for i in range(0, len(allList)):
            root = os.path.splitext(allList[i])
            inList.append(root[0])
        inList = fnmatch.filter(inList,"*_i")
        for i in range(0, len(inList)):
            inList[i] = inList[i][:-2]
            jPath = self.tdPath+'/'+inList[i]
            quePath,ansPath = ji.judgeImgSize(jPath)
            if(quePath != 'null' and ansPath != 'null'):
                outList.append((quePath,ansPath))
        return outList
    def splitDeta(self,quePath,ansPath,trainQuePath,trainAnsPath,
                  testQuePath,testAnsPath,testLatio):
        self.resetDir(trainQuePath)
        self.resetDir(trainAnsPath)
        self.resetDir(testQuePath)
        self.resetDir(testAnsPath)
        allList = [relpath(x, quePath) for x in glob(join(quePath, '*'))]
        random.shuffle(allList)
        testLength = int(len(allList)*testLatio)
        #テストデータ
        for i in range(0,testLength):
            shutil.move(quePath+'/'+allList[i],testQuePath+'/'+str(i)+'.png')
            shutil.move(ansPath+'/'+allList[i],testAnsPath+'/'+str(i)+'.png')
        for i in range(testLength,len(allList)):
            shutil.move(quePath+'/'+allList[i],trainQuePath+'/'+str(i-testLength)+'.png')
            shutil.move(ansPath+'/'+allList[i],trainAnsPath+'/'+str(i-testLength)+'.png')
        os.rmdir(quePath)
        os.rmdir(ansPath)
"""
listUp = ListUp()
listUp.splitDeta('que','ans','trainQue',
                 'trainAns','testQue','testAns',0.2)
"""