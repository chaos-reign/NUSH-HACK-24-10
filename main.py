import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

test = 0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


bohan = cv2.imread(r'reference/bohan.jpg')
zeqi = cv2.imread(r'reference/zeqi.jpg')
kyan = cv2.imread(r'reference/kyan.jpg')

def detect_faces(image):
    # Load a pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(0,0))
    
    return faces, gray

def extract_orb_features(image, face_region):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Crop the face region from the image
    x, y, w, h = face_region
    face_image = image[y:y+h, x:x+w]
    
    # Detect ORB keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(face_image, None)
    
    return keypoints, descriptors, face_image

def match_faces(image1, image2):
    # Detect faces in both images
    faces1, gray1 = detect_faces(image1)
    faces2, gray2 = detect_faces(image2)
    
    # Initialize the Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    best_match = None
    max_matches = 0
    
    # Loop through all faces in the first image
    for face1 in faces1:
        kp1, des1, face_image1 = extract_orb_features(image1, face1)
        
        # Loop through all faces in the second image
        for face2 in faces2:
            kp2, des2, face_image2 = extract_orb_features(image2, face2)
            
            # Match the descriptors using BFMatcher
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                
                # Sort matches based on distance (lower distance is better)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Keep track of the best match (if this one has more matches)
                if len(matches) > max_matches:
                    max_matches = len(matches)
                    best_match = (face_image1, face_image2, matches)
    
    return best_match

def skibidi(frame):

    person_with_best_match = []
    bohan_num_matches = 0
    kyan_num_matches = 0
    zeqi_num_matches = 0
    # Convert the frame to grayscale (Haar Cascade works on grayscale images)
    faces, gray = detect_faces(frame)



    # Display the resulting frame with detected faces

    frame_faces, frame_gray = detect_faces(frame)

    bohan_faces, bohan_Gray = detect_faces(bohan) 
    kyan_faces, kyan_Gray = detect_faces(kyan) 
    zeqi_faces, zeqi_Gray = detect_faces(zeqi) 

    # Initialize the Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    best_match = None
    max_matches = 0
    
    # Loop through all faces in the first image
    for fidx in range(len(frame_faces)):
        frame_face = frame_faces[fidx] 
        kp1, des1, frame_face_image = extract_orb_features(frame, frame_face)
        
        # Loop through all faces in the second image
        bohan_num_matches = 0
        kyan_num_matches = 0
        zeqi_num_matches = 0

        # Loop through all faces in the second image
        for kyan_face in kyan_faces:
            kp2, des2, kyan_face_image = extract_orb_features(kyan, kyan_face) 

            # Match the descriptors using BFMatcher
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                
                # Sort matches based on distance (lower distance is better)
                matches = sorted(matches, key=lambda x: x.distance)
                
                kyan_num_matches = len(matches)


        for bohan_face in bohan_faces:
            kp2, des2, bohan_face_image = extract_orb_features(bohan, bohan_face) 

            # Match the descriptors using BFMatcher
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                
                # Sort matches based on distance (lower distance is better)
                matches = sorted(matches, key=lambda x: x.distance)
                
                bohan_num_matches = len(matches)



        # Loop through all faces in the second image
        for zeqi_face in zeqi_faces:
            kp2, des2, zeqi_face_image = extract_orb_features(zeqi, zeqi_face)

            # Match the descriptors using BFMatcher
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)

                # Sort matches based on distance (lower distance is better)
                matches = sorted(matches, key=lambda x: x.distance)

                zeqi_num_matches = len(matches)


    # decide who 
    highest = max(zeqi_num_matches,kyan_num_matches,bohan_num_matches)
    if highest > 15:
        level = highest - 10
    else:
        level = 100
    if zeqi_num_matches > level:
        person_with_best_match.append('Zeqi')
    if kyan_num_matches > level:
        person_with_best_match.append('Kyan')
    if bohan_num_matches > level:
        person_with_best_match.append('Bohan')
    print(kyan_num_matches)
    print(bohan_num_matches)
    print(zeqi_num_matches)
            
    try: 
        return person_with_best_match
    except Exception as e:
        print(str(e))
        return ["An Error has occurred, please retake the Photo"]
    
        


# Function to save the image to a file
def save_image(image, filename):
    image.save(filename)

# Streamlit App interface
st.title("AI Attendance Taker")
st.write("Click the button below to take a photo")
st.write("It will be automatically proccessed to give you who is present in picture")

# Place the camera widget in the center of the page
camera_image = st.camera_input("Take a photo")

# If an image is captured, display and save it
if camera_image:
    # Convert the Streamlit Image object to a PIL Image
    img = np.array(Image.open(camera_image))
    sigma = skibidi(img)
    sigma = ", ".join(sigma)
    st.write(f"These people are detected in frame: {sigma}")
else:
    st.info("Capture a photo to see the result.")




    





    
    