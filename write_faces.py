import face_recognition
import glob
import os
file = open("known_faces.txt", "w")
i=0
img = []
img_encoding = []
known_face_encodings = []
for f in glob.glob(os.path.join("./faces/*.jpg")):
    img.append(face_recognition.load_image_file("./faces/img{}.jpg".format(i+1)))
    img_encoding.append(face_recognition.face_encodings(img[i])[0]) 
    #file.write(str(img_encoding[i]))
    i = i+1
file.close()
