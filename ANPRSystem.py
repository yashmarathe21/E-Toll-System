import tkinter
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import tkinter as tk, threading
import imageio
from PIL import Image, ImageTk
import PIL
import numpy as np
from copy import deepcopy
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2
import pyrebase

from firebase import firebase
firebase = firebase.FirebaseApplication('https://project03-c99a3.firebaseio.com/', None)

config = {
        "apiKey": "AIzaSyB-qkCth4Oj0K8ZPpAjFnOybmzcaR59tIU",
        "authDomain": "project03-c99a3.firebaseapp.com",
        "databaseURL": "https://project03-c99a3.firebaseio.com",
        "projectId": "project03-c99a3",
        "storageBucket": "project03-c99a3.appspot.com",
        "messagingSenderId": "230381964515",
        "appId": "1:230381964515:web:758a4ba2c22365e3ec09d6"
}

firebase_img = pyrebase.initialize_app(config)
storage = firebase_img.storage()
storage.child()

def video():
    root.filename = filedialog.askopenfilename(initialdir="C:/Users/DELL/PycharmProjects/Automated_EToll_System",
                                               title="Select file",
                                               filetypes=(("video files", "*.mp4"),
                                                          ("all files", "*.*")))
    video_name = root.filename
    video = imageio.get_reader(video_name)

    def stream(label):
        for image in video.iter_data():
            frame_image = ImageTk.PhotoImage(Image.fromarray(image))
            label.config(image=frame_image)
            label.image = frame_image
            label.place(x=50, y=175, height=450, width=700)
    my_label = tk.Label(root)
    my_label.pack()
    thread = threading.Thread(target=stream, args=(my_label,))
    thread.daemon = 2
    thread.start()

    def process():

        def rotateImage(image, angle):
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

        def FrameCapture(path):
            # Path to video file
            vidObj = cv2.VideoCapture(path)
            # Used as counter variable
            count = 0
            # checks whether frames were extracted
            success = 1
            while success:
                # vidObj object calls read
                # function extract frames
                success, image = vidObj.read()
                # Saves the frames with frame-count
                if (success):
                    if(image.shape==(480,640,3)):
                        image = rotateImage(image, -90)
                    cv2.imwrite("frame%d.jpg" % count, image)
                    count += 1
                else:
                    return count
            return count

        def preprocess(img):
            """This function takes an image, applies blurring, uses sobel
            to get horizontal lines. It then returns the binarized image"""
            # cv2.imshow("Input",img)
            imgBlurred = cv2.GaussianBlur(img, (5, 5), 0)
            gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            # cv2.imshow("Sobel",sobelx)
            ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow("Threshold",threshold_img)
            return threshold_img

        def cleanPlate(plate):
            """This function gets the countours that most likely resemeber the shape
            of a license plate"""
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            thresh = cv2.dilate(gray, kernel, iterations=1)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                max_cnt = contours[max_index]
                max_cntArea = areas[max_index]
                x, y, w, h = cv2.boundingRect(max_cnt)
                if not ratioCheck(max_cntArea, w, h):
                    return plate, None
                cleaned_final = thresh[y:y + h, x:x + w]
                #cv2.imshow("Function Test", cleaned_final)
                return cleaned_final, [x, y, w, h]
            else:
                return plate, None

        def extract_contours(threshold_img):
            """This function returns the extracted contours"""
            element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
            morph_img_threshold = threshold_img.copy()
            cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
            #cv2.imshow("Morphed", morph_img_threshold)
            contours, hierarchy = cv2.findContours(morph_img_threshold, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_NONE)
            return contours

        def ratioCheck(area, width, height):
            """This function inspects the ratio of the contour to ensure it meets the requirements
            suitable to a real license plate"""
            ratio = float(width) / float(height)
            if ratio < 1:
                ratio = 1 / ratio
            aspect = 4.7272
            min = 15 * aspect * 15  # minimum area
            max = 125 * aspect * 125  # maximum area
            rmin = 3
            rmax = 6
            if (area < min or area > max) or (ratio < rmin or ratio > rmax):
                return False
            return True

        def isMaxWhite(plate):
            """Checks the average color of the potential plate and if there is more
            white than black colors it returns true"""
            avg = np.mean(plate)
            if (avg >= 115):
                return True
            else:
                return False

        def validateRotationAndRatio(rect):
            """Checks the angle of the rectangle potential license plate"""
            (x, y), (width, height), rect_angle = rect
            if (width > height):
                angle = -rect_angle
            else:
                angle = 90 + rect_angle
            if angle > 15:
                return False
            if height == 0 or width == 0:
                return False
            area = height * width
            if not ratioCheck(area, width, height):
                return False
            else:
                return True

        def deduct(text):
            factor = 50
            s = 'vehicles/'+str(text)
            s_money = s + '/money'
            money = firebase.get(s_money, '')
            firebase.put(s, 'money', money-factor)
            
        def check_vehicle(text):
            s = 'vehicles/' + str(text)
            s_money = s + '/money'
            if(s_money is None):
                return False
            return True

        def cleanAndRead(img, contours):
            """Takes the extracted contours and once it passes the rotation
            and ratio checks it passes the potential license plate to PyTesseract for OCR reading"""
            for i, cnt in enumerate(contours):
                min_rect = cv2.minAreaRect(cnt)
                if validateRotationAndRatio(min_rect):
                    x, y, w, h = cv2.boundingRect(cnt)
                    plate_img = img[y:y + h, x:x + w]
                    if (isMaxWhite(plate_img)):
                        clean_plate, rect = cleanPlate(plate_img)
                        if rect:
                            x1, y1, w1, h1 = rect
                            x, y, w, h = x + x1, y + y1, w1, h1
                            #cv2.imshow("Cleaned Plate", clean_plate)
                            #cv2.waitKey(0)
                            plate_im = Image.fromarray(clean_plate)
                            text = tess.image_to_string(plate_im, lang='eng')
                            print(text)
                            # print("Detected Text : ", text)
                            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            upload_img = img
                            text = ''.join(j for j in text if j.isalnum())
                            # print("Cleaned text : ", text)
                            #cv2.imshow("Detected Plate", img)
                            #cv2.waitKey(0)
                            return text

        cnt = FrameCapture(video_name)
        # print(cnt)
        # print('Number of frames generated', cnt)
        print("DETECTING PLATE . . .")
        for i in range(cnt - 1, 0, -1):
            # for i in range(0,cnt):
            # Path to the license plate you wish to read
            img = "frame" + str(i) + ".jpg"
            img = cv2.imread(img)
            scale_percent = 60  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            threshold_img = preprocess(img)
            contours = extract_contours(threshold_img)
            text = cleanAndRead(img, contours)

            if (not text):
                # print("No text detected")
                continue
            print(text)

            if (text[2] == 'O'):
                lst = list(text)
                lst[2] = '0'
                text = ''.join(map(str, lst))
            if (text[3] == 'O'):
                lst = list(text)
                lst[3] = '0'
                text = ''.join(map(str, lst))
            if (text[4] == '0'):
                lst = list(text)
                lst[4] = 'O'
                text = ''.join(map(str, lst))
            if (text[5] == '0'):
                lst = list(text)
                lst[5] = 'O'
                text = ''.join(map(str, lst))
            print(text)
            cnt_dig = 0
            cnt_chars = 0
            for j in text:
                if (j.isdigit()):
                    cnt_dig += 1
                else:
                    cnt_chars += 1
            ch = text[0:2]
            rto_codes = ['AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
                         'HR', 'HP', 'JH', 'JK', 'KA', 'KL', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ',
                         'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB']
            if(check_vehicle(text)):
                if (cnt_dig == 6 and (cnt_chars == 3 or cnt_chars == 4)):
                    if (ch in rto_codes):
                        try:
                            lbl = Label(root, text="Detected Number Plate", font=("Times New Roman", 25))
                            lbl.place(x=1000, y=75, height=75, width=350)
                            lbl = Label(root, text=text, font=("Times New Roman", 50))
                            lbl.place(x=950, y=300, height=100, width=450)
                            button7 = Button(root, text="Money deducted from user's account", command=deduct(text))
                            button7.place(x=950, y=700, height=50, width=500)
                            s = text + ".jpg"
                            print(s)
                            cv2.imwrite(text + ".jpg", img)
                            path_on_cloud = s
                            path_local = s
                            storage.child(path_on_cloud).put(path_local)
                            cv2.destroyAllWindows()
                            break
                        except:
                            pass
                    else:
                        # print("No accurate RTO code detected.")
                        continue
                else:
                    # print("Inconsistent number of characters extracted.")
                    continue
            else:
                continue

    button2 = Button(root, text="Find the number plate", command=process)
    button2.place(x=150, y=700, height=50, width=500)

def first_page():
    root.title("ANPR System")
    root.geometry("1200x1000")
    lbl = Label(root, text="ANPR System", font=("Times New Roman", 25))
    lbl.place(x=50, y=30, height=100, width=350)
    button1 = Button(root, text="Select video", command=video)
    button1.place(x=500, y=60, height=25, width=150)

root = Tk()
C = Canvas(root, bg="blue", height=250, width=300)
filename = PhotoImage(file="lock.gif")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
first_page()
root.mainloop()