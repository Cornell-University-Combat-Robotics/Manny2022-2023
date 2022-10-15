// cv2 works only using vnc and logged in as a pi user instead of a firmware user

import cv2

cap = cv2.VideoCapture(0)

def getImg(display=False,size=[480,240]):
  _, img = cap.read()
  img = cv2,resize(img, (size[0],size[1]))
  if display:
    cv2.imshow('IMG',img)
  return img
  
if __name__ == 'main':
while True:
  img = getImg(True)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
