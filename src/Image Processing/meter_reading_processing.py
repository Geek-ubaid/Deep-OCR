
# coding: utf-8

import os
import cv2
import numpy as np
from PIL import Image
import requests
import imutils
import json

path = r"C:\Users\HP\Desktop\Projects\Tarp Project\RAW IMAGES\1)Day Light\\"

files = [file for file in os.listdir(path) if file.endswith('.jpg')]

print(sorted(files))


def meter_disp_segment(img_path):

    imgArr_o = cv2.imread(path + img_path)
    imgArr = cv2.cvtColor(imgArr_o, cv2.COLOR_BGR2HSV)
    roi_lower = np.array([40, 25, 0])
    roi_upper = np.array( [80, 255, 255])
    mask = cv2.inRange(imgArr, roi_lower, roi_upper)
    
    imgArr = cv2.bitwise_and(imgArr_o,imgArr_o, mask= mask)
    
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        wbuffer = 0.75 * w
        hbuffer = 0.1 * h
        imgArr_ext = imgArr_o[y:y + h + int(hbuffer), x:x + w + int(wbuffer)]
        
        imgArr_ext_gray = cv2.cvtColor(imgArr_ext, cv2.COLOR_BGR2GRAY)
        imgArr_ext_pp = cv2.adaptiveThreshold(imgArr_ext_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
        imgArr_ext_pp = cv2.medianBlur(imgArr_ext_pp, 13)
        cv2.rectangle(imgArr_o, (x, y), (x + w + int(wbuffer), y + h + int(hbuffer)), (255, 0, 255), 10)
        
        break

  
    cv2.imwrite(img_path.split('.')[0] + '_ext.jpg', imgArr_ext)
    cv2.imwrite(img_path.split('.')[0] + '_mask.jpg', mask)
    cv2.imwrite(img_path.split('.')[0] + '_bb.jpg', imgArr_o)
    cv2.imwrite(img_path.split('.')[0] + '_pp.jpg', imgArr_ext_pp)
    print("Processing of " + img_path + '--> DONE')


for meter in files:
    meter_disp_segment(meter)



def ocr_space_file(filename, overlay=True, api_key='2e82c5d28988957', language='eng'):

	payload = {'isOverlayRequired': overlay,
	           'apikey': api_key,
	           'language': language,
	           'OCREngine':1
	           }
	with open(filename, 'rb') as f:
	    r = requests.post('https://api.ocr.space/parse/image',
	                      files={filename: f},
	                      data=payload,
	                      )

	result = r.content.decode()
	result = json.loads(result) 
	parsed_results = result.get("ParsedResults")[0]
	text_detected = parsed_results.get("ParsedText")
	print(text_detected)





# Use examples:
ocr_space_file(filename=r'C:\Users\HP\Desktop\Projects\Tarp Project\RAW IMAGES\1)Day Light\50.jpg', language='pol')
# test_url = ocr_space_url(url='http://i.imgur.com/31d5L5y.jpg')

    	