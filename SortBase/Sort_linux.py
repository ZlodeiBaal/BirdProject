import cv2
import os
import shutil


CurrentBase = "/home/anna/Public/RaspberryPi/base_3/"
StartSaveIndex=703;
SaveBase="/home/anna/Public/RaspberryPi/FinalBase/"
LastImg = ""
LastTxt = ""

backspace=1113864 #BackSpaceCHR
#Types of birds:
# 1048684 - L / Lasorevka - Cyanistes caeruleus
# 1048691 - S / Sinitsa - Parus major
# 1048608 - Space - Empty
BirdTypes = [1048608, 1048684,1048691]

#Types of quality
# 1 - 1048625
# 2 - +1
# 3 - +1
# 4 - +1
# 5 - +1
# 6 - +1
# 7 - +1
# 8 - +1
# 9 - +1
QualTypes = [1048625,1048626,1048627,1048628,1048629,1048630,1048631,1048632,1048633]

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
