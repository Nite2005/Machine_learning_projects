# write a python script that captures images from your webcam video stream
# Extracts all faces the image frame (using haarcascades)
# stores the Face information into numpy arrays

# 1.Read and show video stream , capture images
# 2.Detect Faces and show bounding box (haarcascades)
# 3.flatten the largest face image (gray sale) and save in a numpy array 
# 4.Reapeat the above for multiple people to generate training data




# import statements
import cv2
import numpy as np


# Init web cam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

#face detection 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt.xml")

face_data=[]
dataset_path="./data/"
file_name= input("Enter the name of the person: ")
# Check if the cascade was loaded successfully
if face_cascade.empty():
    print("Error: Could not load face cascade.")
    exit()

while True:
    ret, frame = cap.read()
    if ret==False:
        continue
    
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)

    if len(faces)==0:
        continue

    faces = sorted(faces,key=lambda f:f[2]*f[3])

    #pick the last face (because it has largest area)
    for face in faces[-1:]:
        #draw bounding box or the rectangle

        x,y,w,h = face
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        #extract (crop out the required face ) : region of interact 
        offset = 10 
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        face_section = face_section.flatten()
        face_data.append(face_section)
        print(len(face_section))

    # cv2.imshow("Frame",frame)
    cv2.imshow("GrayFrame",gray_frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


#convert face data list into numpy array
face_data = np.array(face_data)
# face_data = face.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save this data into file system 
np.save(dataset_path+file_name+'.npy',face_data)
print("Data saved successfully!!:)")

cap.release()
cv2.destroyAllWindows()


