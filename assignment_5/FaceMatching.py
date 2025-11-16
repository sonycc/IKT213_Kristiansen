import face_recognition
import os
import numpy as np

# ----------------------
# CONFIG
# ----------------------
known_dir = "./known_people"
unknown_dir = "./unknown_people"
tolerance = 0.54  # Change this to whatever tolerance you want

# ----------------------
# LOAD KNOWN FACES
# ----------------------
known_encodings = []
known_names = []

for filename in os.listdir(known_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(known_dir, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)

# ----------------------
# CHECK UNKNOWN FACES
# ----------------------
for filename in os.listdir(unknown_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(unknown_dir, filename)
        unknown_image = face_recognition.load_image_file(img_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        for i, unknown_encoding in enumerate(unknown_encodings):
            distances = face_recognition.face_distance(known_encodings, unknown_encoding)
            best_index = np.argmin(distances)
            if distances[best_index] <= tolerance:
                match_name = known_names[best_index]
            else:
                match_name = "Unknown"

            print(f"Found {match_name} in {filename} with distance {distances[best_index]:.3f}")