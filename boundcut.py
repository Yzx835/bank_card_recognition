import cv2 as cv
import copy


def nothing(x):
    pass


src = cv.imread("test_images/2.jpeg")


gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
rows = gray.shape[0]
cols = gray.shape[1]

lineal = copy.deepcopy(gray)
flat = gray.reshape((cols * rows,)).tolist()
A = min(flat)
B = max(flat)
for i in range(rows):
    for j in range(cols):
        lineal[i][j] = 255 * (gray[i][j] - A) / (B - A)

blur = cv.GaussianBlur(lineal, (5, 5), 0)

canny = cv.Canny(blur, 30, 60)


canny = cv.dilate(canny, None)
#canny = cv.erode(canny, None)

cont, hie = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

area = []
for cnt in cont:
    area.append(cv.arcLength(cnt, True))

maxin = area.index(max(area))

cont[maxin] = cv.approxPolyDP(cont[maxin], 5, True)
x, y, w, h = cv.boundingRect(cont[maxin])


cv.drawContours(src, cont, area.index(max(area)), (0,100,0), 2)

cv.imshow('src', src[y:y+h, x:x+w])
cv.imshow('lineal', lineal)
cv.imshow('canny', canny)


cv.waitKey(0)
cv.destroyAllWindows()

'''
cv.namedWindow("res")
cv.createTrackbar('min','res', 0, 255, nothing)
cv.createTrackbar('max','res', 1, 5, nothing)
while(1):
    if cv.waitKey(1)&0xFF==27:
        break
    maxVal = cv.getTrackbarPos('max', 'res')
    minVal = cv.getTrackbarPos('min', 'res')
    canny = cv.Canny(blur, minVal, minVal * maxVal)
    cv.imshow('res', canny)
'''
