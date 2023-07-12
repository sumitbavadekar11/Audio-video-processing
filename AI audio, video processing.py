import cv2

def count_speakers(video_path):
    
    video = cv2.VideoCapture(video_path)

    speaker_count = 0
    previous_frame_faces = []

    while True:
        
        ret, frame = video.read()

        if not ret:
            break

       
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        
        if len(faces) > len(previous_frame_faces):
            speaker_count += 1

        
        previous_frame_faces = faces

    video.release()

    return speaker_count

def detect_faces_and_images(video_path):
    
    video = cv2.VideoCapture(video_path)

    while True:
        
        ret, frame = video.read()
        if not ret:
            break

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


video_path = 'D:\\Data Science\\NLP\\New folder\\Video.mp4'

speaker_count = count_speakers(video_path)
print("Number of speakers in the video:", speaker_count)


detect_faces_and_images(video_path)
