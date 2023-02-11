import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import xgboost
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

crop_dir = "C:/Users/Owner/Prac/Crop_recommendation.csv"
cdf = pd.read_csv(crop_dir)
# loading in the model to predict on the data
pickle_in = open('classifier_bihar.pkl', 'rb')
classifier_bihar = pickle.load(pickle_in)
pickle_in = open('classifier_jharkhand.pkl', 'rb')
classifier_jharkhand = pickle.load(pickle_in)
pickle_in = open('classifier_madhya_pradesh.pkl', 'rb')
classifier_madhya_pradesh = pickle.load(pickle_in)
pickle_in = open('classifier_maharashtra.pkl', 'rb')
classifier_maharashtra = pickle.load(pickle_in)
pickle_in = open('classifier_rajasthan.pkl', 'rb')
classifier_rajasthan = pickle.load(pickle_in)
pickle_in = open('classifier_uttar_pradesh.pkl', 'rb')
classifier_uttar_pradesh = pickle.load(pickle_in)
pickle_in = open('classifier_uttarkhand.pkl', 'rb')
classifier_uttarakhand = pickle.load(pickle_in)

pickle_in = open('classifier_final.pkl', 'rb')
classifier_final = pickle.load(pickle_in)

le = preprocessing.LabelEncoder()
# X = cdf.iloc[:, 0:7]
y = cdf['label']
y = le.fit_transform(y)
# # print(y.shape)
# y = y.reshape(-1,1)
# # print(y)
# temp=[]
# for i in range(len(y)):
# 	temp.append(y[i][0])
# # print(temp)
# y=temp
# # print(y[0])


def welcome():
    return 'welcome all'

# defining the function which will make the rainfall_prediction using
# the data which the user inputs


def rainfall_prediction(state, months, year):
    months = months.lower()
    state = state.upper()
    months_list = ['january', 'february', 'march', 'april', 'may', 'june',
                   'july', 'august', 'september', 'october', 'november', 'december']
    l = [0]*12
    l[months_list.index(months)] = 1
    l.append(int(year))
    test = []
    test.append(l)
    test = pd.DataFrame(data=test, columns=["level_0_1", "level_0_2", "level_0_3", "level_0_4", "level_0_5",
                        "level_0_6", "level_0_7", "level_0_8", "level_0_9", "level_0_10", "level_0_11", "level_0_12", "YEAR"])
    if state == "BIHAR":
        return classifier_bihar.predict(test)
    elif state == "JHARKHAND":
        return classifier_jharkhand.predict(test)
    elif state == "MADHYA PRADESH":
        return classifier_madhya_pradesh.predict(test)
    elif state == "MAHARASHTRA":
        return classifier_maharashtra.predict(test)
    elif state == "RAJASTHAN":
        return classifier_rajasthan.predict(test)
    elif state == "UTTAR PRADESH":
        return classifier_uttar_pradesh.predict(test)
    elif state == "UTTARAKHAND":
        return classifier_uttarakhand.predict(test)
    else:
        return "Invalid State"
  


def crop_suggestion(Nitrogen, Phosphorus, Potassium, temperature, humidity, ph, rainfall):
    l = [float(Nitrogen), float(Phosphorus), float(Potassium), float(
        temperature), float(humidity), float(ph), float(rainfall)]
    test = [l]
    test = pd.DataFrame(data=test, columns=[
                        "N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    crop = classifier_final.predict(test)

    return crop
# this is the main function in which we define our webpage


def main():
    # giving the webpage a title
    # st.title("Rainfall rainfall_Prediction")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Krushi Mitra </h1>
	</div>
	"""

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)

    # the following lines create text boxes in which the user can enter
    # the data required to make the rainfall_prediction
    state = st.text_input("State [As of now data only of Bihar, Jharkhand, Madhya Pradesh, Maharashtra, Rajasthan, Uttar Pradesh and Uttarkhand is available]", "")
    month = st.text_input("Month", "")
    year = st.text_input("Year", "")
    Nitrogen = st.text_input("Nitrogen content in soil", "")
    Phosphorus = st.text_input("Phosphorus content in soil", "")
    Potassium = st.text_input("Potassium content in soil", "")
    temperature = st.text_input("Temperature in the region", "")
    humidity = st.text_input("Humidity in the region", "")
    ph = st.text_input("pH level of soil", "")

    result = ""
    ans = ""

    # the below line ensures that when the button called 'Predict' is clicked,
    # the rainfall_prediction function defined above is called to make the rainfall_prediction
    # and store it in the variable result
    if st.button("Predict"):
        result = rainfall_prediction(state, month, year)
        ans = crop_suggestion(Nitrogen, Phosphorus, Potassium,
                              temperature, humidity, ph, result[0])
    ans = le.inverse_transform(ans)[0]

    st.success('The suggested crop is   {}'.format(ans))


if __name__ == '__main__':
    main()
