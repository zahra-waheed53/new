import cv2

image = cv2.imread('image.jpg')

# Extracting the height and width of an image
h, w = image.shape[:2]
print(f"height: {h}, width: {w}")


#  Extracting RGB values.
B, G, R = image[100, 100]
# print(f'B: {B}, G: {G}, R: {R}')


#calculate the region of interest by slicing the pixels of the image
roi = image[100 : 300, 200 : 500]
# cv2.imshow("ROI", roi)
# cv2.waitKey(0)


# Resizing the Image
resized = cv2.resize(image, (200, 500))
h, w = resized.shape[:2]
print(f"height: {h}, width: {w}")
# cv2.imshow('resized', resized)
# cv2.waitKey(0)

# maintain a proper aspect ratio.
ratio = 800 / w
dim = (800, int(ratio * h))
resize_aspect = cv2.resize(image, dim)
# cv2.imshow('resize_aspect', resize_aspect)
# cv2.waitKey(0)


output = image.copy()
rectangle_on_image = cv2.rectangle(output, (100, 100), (300, 300), (255,0, 100), 5)
# cv2.imshow("rectangle", rectangle_on_image)
# cv2.waitKey(0)

output = image.copy()
text_on_image = cv2.putText(output, "Demo", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 100, 100), 3)
# cv2.imshow("Text", text_on_image)
# cv2.waitKey(0)
