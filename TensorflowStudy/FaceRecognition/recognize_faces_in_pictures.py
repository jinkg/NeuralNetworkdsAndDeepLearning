import face_recognition


def find(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    destance = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    the_best = min(min(destance), tolerance)
    return list(destance <= the_best)


# Load the jpg files into numpy arrays
biden_image = face_recognition.load_image_file("img/biden.jpg")
obama_image = face_recognition.load_image_file("img/obama.jpg")
shasha_image = face_recognition.load_image_file("img/shasha5.jpg")
yalin_image = face_recognition.load_image_file("img/yalin4.png")

unknown_images = [face_recognition.load_image_file("img/obama2.jpg"),
                  face_recognition.load_image_file("img/shasha2.jpg"),
                  face_recognition.load_image_file("img/shasha3.jpg"),
                  face_recognition.load_image_file("img/shasha4.jpg"),
                  face_recognition.load_image_file("img/shasha5.jpg"),
                  face_recognition.load_image_file("img/yalin1.jpg"),
                  face_recognition.load_image_file("img/yalin2.jpg"),
                  face_recognition.load_image_file("img/yalin3.jpg"),
                  face_recognition.load_image_file("img/unknown1.png")]

# unknown_image = face_recognition.load_image_file("img/yalinshasha1.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encordings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
yalin_face_encoding = face_recognition.face_encodings(yalin_image)[0]
shasha_face_encoding = face_recognition.face_encodings(shasha_image)[0]

unknown_face_encodings = [face_recognition.face_encodings(unknown_image)[0] for unknown_image in unknown_images]
# unknown_face_encoding = face_recognition.face_encodings(unknown_image)[1]

known_faces = [
    biden_face_encoding,
    obama_face_encoding,
    yalin_face_encoding,
    shasha_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
# results = [face_recognition.compare_faces(known_faces, unknown_face_encoding)
#            for unknown_face_encoding in unknown_face_encodings]

results = [find(known_faces, unknown_face_encoding)
           for unknown_face_encoding in unknown_face_encodings]

# result = find(known_faces, unknown_face_encoding)
# print("Is the unknown face a picture of Biden? {}".format(result[0]))
# print("Is the unknown face a picture of Obama? {}".format(result[1]))
# print("Is the unknown face a picture of Yalin? {}".format(result[2]))
# print("Is the unknown face a picture of Shasha? {}".format(result[3]))
# print("Is the unknown face a new person that we've never seen before? {}".format(True not in result))
# print("--------")

for result in results:
    print("Is the unknown face a picture of Biden? {}".format(result[0]))
    print("Is the unknown face a picture of Obama? {}".format(result[1]))
    print("Is the unknown face a picture of Yalin? {}".format(result[2]))
    print("Is the unknown face a picture of Shasha? {}".format(result[3]))
    print("Is the unknown face a new person that we've never seen before? {}".format(True not in result))
    print("--------")
