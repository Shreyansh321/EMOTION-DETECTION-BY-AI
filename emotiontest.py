from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import streamlit as st
import base64


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

classifier =load_model('Emotion_little_vgg.h5')
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']


frameST = st.empty()
st.sidebar.title("Detect Options")
detect=st.sidebar.button('Emotion Detection')
end = st.sidebar.button('End Session')

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('robo1.0.jpeg')



from load_css import local_css

local_css("styles.css")
 
t = "<div class='bold highlight'>Facial Emotion Detection</div>"

st.markdown(t, unsafe_allow_html=True)



def recog():
   cap = cv2.VideoCapture(0)  
   while True:
    # Grab a single frame of video
      ret, frame = cap.read()
      labels = []
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      faces = face_classifier.detectMultiScale(gray,1.3,5)

      for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    #cv2.imshow('Emotion Detector',frame)
      frameST.image(frame, channels="BGR")
      if end:
          break


   cap.release()
   cv2.destroyAllWindows()
        

if detect:
    recog()
   


