from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import os

# Initialize FastAPI app
app = FastAPI()

# Define paths for models and datasets
model_dir = "model/"
decision_tree_model_path = os.path.join(model_dir, "decision_tree_model.pkl")
random_forest_model_path = os.path.join(model_dir, "random_forest_model.pkl")
naive_bayes_model_path = os.path.join(model_dir, "naive_bayes_model.pkl")
stud_training_csv_path = os.path.join(model_dir, "stud_training.csv")

# Load the models
clf3 = joblib.load(decision_tree_model_path)
clf4 = joblib.load(random_forest_model_path)
gnb = joblib.load(naive_bayes_model_path)

# Load the dataset
df = pd.read_csv(stud_training_csv_path)

# Define lists for interests and courses
l1 = ['Drawing', 'Dancing', 'Singing', 'Sports', 'Video Game', 'Acting', 'Travelling', 'Gardening', 'Animals',
      'Photography', 'Teaching', 'Exercise', 'Coding', 'Electricity Components', 'Mechanic Parts', 'Computer Parts',
      'Researching', 'Architecture', 'Historic Collection', 'Botany', 'Zoology', 'Physics', 'Accounting', 'Economics',
      'Sociology', 'Geography', 'Psycology', 'History', 'Science', 'Bussiness Education', 'Chemistry', 'Mathematics',
      'Biology', 'Makeup', 'Designing', 'Content writing', 'Crafting', 'Literature', 'Reading', 'Cartooning',
      'Debating', 'Asrtology', 'Hindi', 'French', 'English', 'Other Language', 'Solving Puzzles', 'Gymnastics', 'Yoga',
      'Engeeniering', 'Doctor', 'Pharmisist', 'Cycling', 'Knitting', 'Director', 'Journalism', 'Bussiness',
      'Listening Music']

Course = ['BBA- Bachelor of Business Administration', 'BEM- Bachelor of Event Management', 'Integrated Law Course- BA + LL.B',
          'BJMC- Bachelor of Journalism and Mass Communication', 'BFD- Bachelor of Fashion Designing',
          'BBS- Bachelor of Business Studies', 'BTTM- Bachelor of Travel and Tourism Management', 'BVA- Bachelor of Visual Arts',
          'BA in History', 'B.Arch- Bachelor of Architecture', 'BCA- Bachelor of Computer Applications', 'B.Sc.- Information Technology',
          'B.Sc- Nursing', 'BPharma- Bachelor of Pharmacy', 'BDS- Bachelor of Dental Surgery', 'Animation, Graphics and Multimedia',
          'B.Sc- Applied Geology', 'B.Sc.- Physics', 'B.Sc. Chemistry', 'B.Sc. Mathematics', 'B.Tech.-Civil Engineering',
          'B.Tech.-Computer Science and Engineering', 'B.Tech.-Electrical and Electronics Engineering',
          'B.Tech.-Electronics and Communication Engineering', 'B.Tech.-Mechanical Engineering', 'B.Com- Bachelor of Commerce',
          'BA in Economics', 'CA- Chartered Accountancy', 'CS- Company Secretary', 'Diploma in Dramatic Arts', 'MBBS',
          'Civil Services', 'BA in English', 'BA in Hindi', 'B.Ed.']

# Define Pydantic model for the input data
class InterestsInput(BaseModel):
    interests: list

# Prediction function
def predict_course(interests, clf):
    input_vector = [0] * len(l1)
    for interest in interests:
        if interest in l1:
            input_vector[l1.index(interest)] = 1

    prediction = clf.predict([input_vector])
    predicted_course = Course[prediction[0]]
    return predicted_course

# FastAPI endpoint to predict course
@app.post("/predict/")
def predict(input_data: InterestsInput):
    interests = input_data.interests
    decision_tree_course = predict_course(interests, clf3)
    random_forest_course = predict_course(interests, clf4)
    naive_bayes_course = predict_course(interests, gnb)
    return {
        "Decision Tree": decision_tree_course,
        "Random Forest": random_forest_course,
        "Naive Bayes": naive_bayes_course
    }
