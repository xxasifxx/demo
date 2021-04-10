from pickle import EMPTY_LIST
import dlib
import face_recognition
import cv2
import numpy as np
import glob
import os

def detect_faces(group):
    path = 'C:/Attendance/demo-main/database'
    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()
    img = dlib.load_rgb_image(group)
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        crop = img[d.top():d.bottom(), d.left():d.right()]
        cv2.imwrite(os.path.join(path, 'd{}.jpg'.format(i)), crop)
    return len(dets)
    
def f(unknown):
    known_face_encoding = []
    file = open("known_faces.npy", "rb")
    for f in glob.glob(os.path.join("./faces/*.jpg")):
        known_face_encoding.append(np.load(file,allow_pickle=True))
    unknown_image = face_recognition.load_image_file(unknown)
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0] 
    results = face_recognition.compare_faces(known_face_encoding, unknown_encoding, tolerance=0.6) #adjust tolerance parameter until results list shows True for only the image numbers that are part of your desired identity
    print(results)
    
def encode_faces():
    file = open("known_faces.npy", "wb")
    i=0
    img = []
    img_encoding = []
    known_face_encodings = []
    print ("before for loop")
    print (glob.glob("./faces"))
    for f in glob.glob(os.path.join("./faces/*.jpg")):
        print ("in for loop ... ")
        print (i)
        img.append(face_recognition.load_image_file(f))
        img_encoding.append(face_recognition.face_encodings(img[i])[0]) 
        np.save(file, img_encoding[i])
        i = i+1
    print ("after for loop")
    file.close()
    return 0

#for reference   
idendities = {
    "Angelina": ["img1.jpg", "img2.jpg", "img4.jpg", "img5.jpg", "img6.jpg", "img7.jpg", "img10.jpg", "img11.jpg"],
    "Scarlett": ["img8.jpg", "img9.jpg", "img47.jpg", "img48.jpg", "img49.jpg", "img50.jpg", "img51.jpg"],
    "Jennifer": ["img3.jpg", "img12.jpg", "img53.jpg", "img54.jpg", "img55.jpg", "img56.jpg"],
    "Mark": ["img13.jpg", "img14.jpg", "img15.jpg", "img57.jpg", "img58.jpg"],
    "Jack": ["img16.jpg", "img17.jpg", "img59.jpg", "img61.jpg", "img62.jpg"],
    "Elon": ["img18.jpg", "img19.jpg"],
    "Jeff": ["img20.jpg", "img21.jpg"],
    "Sundar": ["img24.jpg", "img25.jpg"],
    "Katy": ["img26.jpg", "img27.jpg", "img28.jpg", "img42.jpg", "img43.jpg", "img44.jpg", "img45.jpg", "img46.jpg"],
    "Matt": ["img29.jpg", "img30.jpg", "img31.jpg", "img32.jpg", "img33.jpg"],
    "Leonardo": ["img34.jpg", "img35.jpg", "img36.jpg", "img37.jpg"],
    "George": ["img38.jpg", "img39.jpg", "img40.jpg", "img41.jpg"]
    
}

if __name__ == "__main__":
    k = detect_faces('./input.jpg') #detect individual faces
    #encode_faces() #build up repository of facial identities
    for i in range(k):
        f("./database/d{}.jpg".format(i)) #match face with identity
