from imutils import paths
import face_recognition
import pickle as pkl
import cv2
import os

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("dataset"))
# initialize the list of known encodings and known names

#print(imagePaths)
knownEncodings = []
knownNames = []


for (i, imagePath) in enumerate(imagePaths):
    #reading the images
    print("Processing image {}/{}".format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    #converting them to rgb for face_recognition
    image =cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #detecting faces
    #cnn is slower but accurate, HOG is faster but less accurate
    boxes = face_recognition.face_locations(rgb, model = "cnn")

    #list of 128 numbers which are face landmarks
    encodings = face_recognition.face_encodings(rgb, boxes)

    for e in encodings:
        knownEncodings.append(e)
        knownNames.append(name)

#wriing in the pickle file
print("Serializing encodings....")
data = {"encodings ":knownEncodings,"names": knownNames}
f = open("encodings_file", "wb")
f.write(pkl.dumps(data))
f.close()
