import face_recognition
import cv2
import numpy as np
import glob
import os


def f(unknown):
    img = []
    img_encoding = []
    known_face_encodings = []
    i=0
    for f in glob.glob(os.path.join("./faces/*.jpg")):
        img.append(face_recognition.load_image_file("./faces/img{}.jpg".format(i+1)))
        img_encoding.append(face_recognition.face_encodings(img[i])[0])
        known_face_encodings.append(img_encoding[i])
        i = i+1

    unknown_image = face_recognition.load_image_file(unknown)
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.compare_faces(known_face_encodings, unknown_encoding, tolerance=0.6) #adjust tolerance parameter until results list shows True for only the image numbers that are part of your desired identity

    #for i in range(len(results)):
    #    print(i+1, " ", results[i])
    return results
def test_angie():
    assert f("./faces/img1.jpg") == f("./faces/img2.jpg")
    assert f("./faces/img4.jpg") == f("./faces/img2.jpg")
def test_scar():
    assert f("./faces/img8.jpg") == f("./faces/img47.jpg")

#for reference, not being used for now    
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
