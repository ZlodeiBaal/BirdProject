import cv2
import os
import shutil


CurrentBase = "F:/BIRD/good/fg1"
StartSaveIndex=2201;
SaveBase="F:/BIRD/FinalBase/"
LastImg = ""
LastTxt = ""

backspace=8 #BackSpaceCHR
#Types of birds:
# 108 - L / Lasorevka - Cyanistes caeruleus
# 115 - S / Sinitsa - Parus major
# 32 - Space - Empty
BirdTypes = [32, 108,115]

#Types of quality
# 1 - 49
# 2 - 50
# 3 - 51
# 4 - 52
# 5 - 53
# 6 - 54
# 7 - 55
# 8 - 56
# 9 - 57
QualTypes = [49,50,51,52,53,54,55,56,57]

def GET_LIST(folder):
    Adress = []
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for i in onlyfiles:
        filename, file_extension = os.path.splitext(i)
        if(file_extension=='.jpg'):
            imgadress = os.path.join(folder,i)
            Adress.append(imgadress)
    return Adress

List = GET_LIST(CurrentBase)
for i in range(len(List)):
    img=cv2.imread(List[i])
    cv2.imshow("Test",img)
    k1 = cv2.waitKey()
    print (k1)
    k2=QualTypes[0]
    if (k1!=backspace and k1!=BirdTypes[0]):
        k2 = cv2.waitKey()
        print (k2)
    if (k1==backspace or k2==backspace):
        i=i-2
        StartSaveIndex = StartSaveIndex-1
        os.remove(LastImg)
        os.remove(LastTxt)
    else:
        BType = 0
        QType=0
        for j in range(len(BirdTypes)):
            if (k1==BirdTypes[j]) or (k2==BirdTypes[j]):
                BType=j
        for j in range(len(QualTypes)):
            if (k1==QualTypes[j]) or (k2==QualTypes[j]):
                QType=j
        shutil.copy(List[i],SaveBase+str(StartSaveIndex)+".jpg")
        fs = open(SaveBase+str(StartSaveIndex)+".txt", 'a')
        fs.write(str(BType) + ' ' + str(QType)  + '\n')
        fs.close()
        LastImg = SaveBase+str(StartSaveIndex)+".jpg"
        LastTxt = SaveBase+str(StartSaveIndex)+".txt"
        StartSaveIndex=StartSaveIndex+1