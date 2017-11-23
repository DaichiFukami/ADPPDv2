import listUp
import editImg

tdPath = 'traindata'
quePath =''
ansPath = ''

listUp = listUp.ListUp(tdPath)
fileList = listUp()
listUp = None
edit = editImg.EditImg()
for i in range(0, len(fileList)):
    edit.outImg('in',fileList[i][0])
    edit.outImg('out',fileList[i][1])