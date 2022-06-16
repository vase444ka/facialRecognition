import face_recognition
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from os import listdir
from os.path import join
import json


test_path = './data/recognition_train'
labels = "./data/face_labels.json"
data = json.load(open(labels))

faces = {}
for img in data:
    faces[img["file_upload"][9:]] = img["annotations"][0]["result"]
print(faces.keys())

cnt_diff = 0
cnt_faces = 0

for f in listdir(test_path):
    cnt_faces = cnt_faces + len(faces[f])
    image = face_recognition.load_image_file(join(test_path, f))
    face_locations = face_recognition.face_locations(image)

    diff = abs(len(face_locations) - len(faces[f]))
    cnt_diff = cnt_diff + diff

    if diff > 0:
        plt_img = plt.imread(join(test_path, f))
        plt.imshow(plt_img)
        for face in face_locations:
            plt.gca().add_patch(Rectangle((face[1], face[0]),
                                          face[3] - face[1],
                                          face[2] - face[0],
                                          linewidth=1, edgecolor='r', facecolor='none'))
        plt.show()

print("\nTotal faces: ", cnt_faces, "\nUndetected faces percentage: ", cnt_diff/cnt_faces*100)