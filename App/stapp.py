import cv2
import pandas as pd
import os
from io import StringIO
from pathlib import Path
import streamlit as st
import numpy as np
import time
import torch
from utils.datasets import letterbox
from torchvision import transforms
from torch.autograd import Variable
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import bardapi
import re

# bardapi key
token = 'WwiPoDPj7P_yu6yztDc21z_13IzJjpGSpv4p_6c81CZ792wBDL9GF1TfmC0vWRNPZh6MDA.'

model_option=0
model=None

from pathlib import Path
import torch
from calc import curl_calc, pushup_count, shoulderPress, squat_calc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "yolov7-w6-pose.pt"

weights = torch.load(MODEL_PATH, map_location=device)
pose_model = weights['model'].float().eval()
clf=torch.load(r'model.pth', map_location=torch.device('cpu'))
clf.eval()
if torch.cuda.is_available():
        pose_model.half().to(device)

classes= ['barbell biceps curl',
 'bench press',
 'chest fly machine',
 'deadlift',
 'decline bench press',
 'hammer curl',
 'hip thrust',
 'incline bench press',
 'lat pulldown',
 'lateral raises',
 'leg extension',
 'leg raises',
 'plank',
 'pull up',
 'push up',
 'romanian deadlift',
 'russian twist',
 'shoulder press',
 'squat',
 't bar row',
 'tricep dips',
 'tricep pushdown']

tsfm = transforms.Compose([
    
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        
])

def pred(img, tsfm):
    tensor_img = tsfm(img)

    tensor_img = torch.unsqueeze(tensor_img,0) 
    

    imginput = Variable(tensor_img)

    prediction = clf(imginput)
    
    
    index = prediction.data.numpy().argmax()
    output = classes[index]
    
    return output        
        
        
st.title("Fit G")
st.sidebar.title('Configure')


source = ("Classify Exercise", "Bicep_Curl", "Pushup","Squat","Shoulder Press", "ChatBot")

source_index = st.sidebar.selectbox("Input", range(len(source)), format_func=lambda x: source[x])
st.sidebar.write("####")
output=None

if source_index == 0 :

        label='Submit'
        st.write('See detections in an Image')
        st.sidebar.write("#####")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:

            with st.spinner(text='classifying'):
                
                picture = Image.open(uploaded_file)
                output=pred(picture,tsfm)
            st.sidebar.write("######")
            st.success(f"Exercise classified as {output.upper()}",  icon="✅")

elif source_index == 1:

        st.write('Bicep Curls!')
        st.sidebar.write("#####")
        
        if st.sidebar.button("Start"):
        
            spinnerlabel='Live Video Stream Enabled \n\n Check for a new popup window to see the live video \n\n Press "Q" to stop the stream'
            
            with st.spinner(text=spinnerlabel):
                curl_calc(pose_model)
            
            st.warning('Webcam has stopped')
            
        
                    
elif source_index == 2:
        st.write('Push Ups!')
        st.sidebar.write("#####")
        
        if st.sidebar.button("Start"):
        
            spinnerlabel='Live Video Stream Enabled \n\n Check for a new popup window to see the live video \n\n Press "Q" to stop the stream'
            
            with st.spinner(text=spinnerlabel):
                pushup_count(pose_model)
            
            st.warning('Webcam has stopped')
    
                      
elif source_index == 3:
        st.write('squats!')
        st.sidebar.write("#####")
        
        if st.sidebar.button("Start"):
        
            spinnerlabel='Live Video Stream Enabled \n\n Check for a new popup window to see the live video \n\n Press "Q" to stop the stream'
            
            with st.spinner(text=spinnerlabel):
                squat_calc(pose_model)
            
            st.warning('Webcam has stopped')
            
elif source_index == 4:
        st.write('Shoulder Press!')
        st.sidebar.write("#####")
        
        if st.sidebar.button("Start"):
        
            spinnerlabel='Live Video Stream Enabled \n\n Check for a new popup window to see the live video \n\n Press "Q" to stop the stream'
            
            with st.spinner(text=spinnerlabel):
                shoulderPress(pose_model)
            
            st.warning('Webcam has stopped')

                
elif source_index == 5:
    
    st.write("Get your queries answered!")
    
    age=st.sidebar.text_input("Age")
    gender=st.sidebar.text_input("Gender")
    height_cms=st.sidebar.text_input("Height in cms")
    working_hours=st.sidebar.text_input("Working Hours")
    workout_time_hours=st.sidebar.text_input("Workout hours")
    body_weight=st.sidebar.text_input("Weight")
    health_condition=st.sidebar.text_input("Any health Conditions")
    
    
    prompts = ("Diet Plan for Bulking", "Sleep Schedule", "Avoiding Injuries","Beginner Weights","Workout Days", "Healthy Fats to Eat")

    prompts_index = st.sidebar.selectbox("Questions", range(len(prompts)), format_func=lambda x: prompts[x])
    st.sidebar.write("####")
    
    diet_plan_bulkin = f"my age is {age} my height is {height_cms} in cms now make me a diet plan for bulking"



    sleep_schedule = f"my age is {age} my working hours  is {working_hours}  now make me a sleep schedule"



    avoid_injuries = f" my age is {age} my workout time is {workout_time_hours}  now tell me how to avoid injuries"


    weight_to_start = f" my age is {age} my body_weight is {body_weight} now suggest me the weight to start with"


    days_of_work = f"person with age {age} and with a health condition of {health_condition} now suggest me how  many days a week should i workout"


    perfect_time = f"a person with job_hours{working_hours} suggest me the better time to workout morning or evening "



    good_fat = f" a person with health condition {health_condition} age {age} and gender{gender} suggest me the best healthy fats to eat "
    
    ls_queries=[diet_plan_bulkin, sleep_schedule, avoid_injuries, weight_to_start, days_of_work, perfect_time]
    
    if st.sidebar.button("Get Answers"):
            
            with st.spinner("Querying"):
                response = bardapi.core.Bard(token).get_answer(ls_queries[prompts_index])

                lines = response['content'].split('\n')
            for i in range(len(lines)):
                st.write(lines[i])
            
                 

