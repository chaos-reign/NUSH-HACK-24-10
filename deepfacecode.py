import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

bohan = cv2.imread('bohan.jpg')
zeqi = cv2.imread('zeqi.jpg')
kyan = cv2.imread('kyan.jpg')

def detect_faces(image):
    # Load a pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
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

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # If frame is not captured correctly, break the loop
    if not ret:
        print("Failed to grab frame.")
        break

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

        for bohan_face in bohan_faces:
            kp2, des2, bohan_face_image = extract_orb_features(bohan, bohan_face) 

            # Match the descriptors using BFMatcher
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                
                # Sort matches based on distance (lower distance is better)
                matches = sorted(matches, key=lambda x: x.distance)
                
                bohan_num_matches = len(matches)

        # Loop through all faces in the second image
        for kyan_face in kyan_faces:
            kp2, des2, kyan_face_image = extract_orb_features(kyan, kyan_face) 

            # Match the descriptors using BFMatcher
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                
                # Sort matches based on distance (lower distance is better)
                matches = sorted(matches, key=lambda x: x.distance)
                
                kyan_num_matches = len(matches)

        # Loop through all faces in the second image
        for zeqi_face in zeqi_faces:
            kp2, des2, zeqi_face_image = extract_orb_features(zeqi, zeqi_face)

            # Match the descriptors using BFMatcher
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)

                # Sort matches based on distance (lower distance is better)
                matches = sorted(matches, key=lambda x: x.distance)

                zeqi_num_matches = len(matches)

        person_with_best_match = 'None'
        # decide who 
        highest = max(zeqi_num_matches, bohan_num_matches, kyan_num_matches)
        if highest > 30:
            if highest == zeqi_num_matches:
                person_with_best_match = 'Zeqi'
            if highest == kyan_num_matches:
                person_with_best_match = 'Kyan'
            if highest == bohan_num_matches:
                person_with_best_match = 'Bohan'
            
        x, y, w, h = faces[fidx]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, person_with_best_match, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        print(f' Zeqi {zeqi_num_matches}')
        print(f' Kyan {kyan_num_matches}')
        print(f' Bohan {bohan_num_matches}')
    
    cv2.imshow("Face Detection", frame)

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()