# Source - https://stackoverflow.com/a/64977522
# Posted by fmw42
# Retrieved 2026-10-28, License - CC BY-SA 4.0
from numpy import average
import cv2
import numpy as np
from skimage import color
from skimage import io

# Source - https://stackoverflow.com/q/64969715
# Posted by Mads, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-28, License - CC BY-SA 4.0

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
# Source - https://stackoverflow.com/a/35967816
# Posted by MSeifert, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-29, License - CC BY-SA 3.0

import operator

def password(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    pixel_values = processor(image, return_tensors="pt").pixel_values  

    generated_ids = model.generate(pixel_values, max_length=20)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Extracted Text: {text}")


def apply_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 60, 50])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    return mask


def delete_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for row in image:
        for line in row:
            line[0] = 0
    cv2.imshow("image", image)
    cv2.waitKey(0)
    

def find_squares(img):
    mask = apply_mask(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 5000:
            rect = cv2.minAreaRect(cnt)
            (w_box, h_box) = rect[1]
            if h_box > 0 and w_box > 0:
                aspect = max(w_box, h_box) / min(w_box, h_box)
                if aspect < 1.35:  # Keep only square-like shapes
                    box = cv2.boxPoints(rect)
                    squares.append(np.int32(box))
    return squares


image = cv2.imread(fr"C:\Users\TLP\-\image processing tools\password detection\normal_pic.jpg")

#delete_color(image)

squares = find_squares(image)

display_img = image.copy()
cv2.drawContours(display_img, squares, -1, (0, 255, 0), 3)
cv2.imshow("image", display_img)
cv2.waitKey(0)

for square in squares:
    dummy = [[square[j][i] for j in range(len(square))] for i in range(len(square[0]))]
    
    crop_img = image[min(dummy[1]):max(dummy[1]), min(dummy[0]):max(dummy[0])]
    h, w = crop_img.shape[:2]
    cropped_images = []
    for i in range(3):
        cropped_images.append(crop_img[int(i*(h * 0.25)):int((i+2)*(h * 0.25)), 0:int(w)])
        cropped_mask = apply_mask(cropped_images[i])
        if(average(cropped_mask) > 100):
            cv2.imshow("cropped", cropped_images[i])
            password(cropped_images[i])