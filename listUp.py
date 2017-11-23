import fnmatch
import os
from os.path import join, relpath
from glob import glob

import judgeImg

class ListUp:
    def __init__(self,tdPath):
        self.tdPath = tdPath
    def __call__(self):
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
lu = ListUp('traindata')

