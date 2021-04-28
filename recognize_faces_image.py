import face_recognition
import pickle
import os
import cv2

#load the pickled file
print("Loading encodings...")
data = pickle.loads(open("encodings_file", "rb").read())
key = list(data.keys())[0]

#face_recognition readable format
image = cv2.imread("examples/example_02.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#encodings for the imput image
print("Recognizing faces...")
boxes = face_recognition.face_locations(rgb, model = "cnn")
encodings = face_recognition.face_encodings(rgb,boxes)

names = []

#compare_faces to compute Euclidian distance b/w input and dataset encodings
for e in encodings:
    matches = face_recognition.compare_faces(data[key], e)
    name = "Unknown"

    #to check if we have a match
    if True in matches:
        #to find indexes of all matched faces
        matchedIndex = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        #to get the name of the matched indexes
        for i in matchedIndex:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        #image with maximum votes is extracted
        name = max(counts, key = counts.get)
    names.append(name)

for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_ITALIC, 0.75, (0, 255,0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)