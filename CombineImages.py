import cv2,numpy as np

def resizeTheImage(bimg,fimg):
    print(bimg.shape[0]/bimg.shape[1],fimg.shape[0]/fimg.shape[1])
    fimg = cv2.resize(fimg, bimg.shape[:2][::-1])
    return fimg


backgorndImage="Images/thumb-1920-82317.jpg"
forgrounfImage="Images/out.png"
fimg=cv2.imread(forgrounfImage,cv2.IMREAD_UNCHANGED)
bimg=cv2.imread(backgorndImage)
fimg=resizeTheImage(bimg,fimg)
print(fimg.shape,bimg.shape)

out=np.zeros(bimg.shape)
print(out.shape)
for i in range(bimg.shape[0]):
    for j in range(bimg.shape[1]):
        if fimg[i,j,3]:
            out[i,j]=fimg[i,j,:3]
        else:
            out[i,j]=bimg[i,j,:]

cv2.imwrite("Images/Gout.jpg",out)


