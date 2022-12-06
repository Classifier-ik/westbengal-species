import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch
import pandas
import pickle
from helpers import predict, upload
from streamlit_option_menu import option_menu

with open('classlabels.pkl', 'rb') as f:
    class_names = pickle.load(f)

st.set_page_config(layout="wide")
st.set_option('deprecation.showfileUploaderEncoding', False)

selected = option_menu(
    menu_title=None,
    options=["Home", "Project"],
    orientation="horizontal",
    default_index=0,
    menu_icon="cast",
    icons=['house-fill', 'gear-fill'],
    styles={
        "nav-link-selected": {"background-color": "#82faa4"},
    }
)

if selected == "Project":
    # st.title("Karam and Ishan's Simple Image Classification App")
    st.write("")
    st.header("Upload an image \U0001F447\U0001F447\U0001F447")
    file_up = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

    if file_up is not None:
        image = Image.open(file_up)

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")

        with col2:
            st.write("Your results are served here...")
            score, bird_name = predict(file_up, class_names)
            # st.write(results)
            if score > 60:
                st.write("Prediction (name): ",
                         bird_name, ",   \nScore: ", score)
            else:
                st.write("No such bird in database!")
                new_bird_name = st.text_input(
                    'Bird Name', '', placeholder="Enter bird name here....")
                if(st.button('Upload this bird to dataset!')):
                    if new_bird_name == "":
                        st.write("Enter bird name!")
                    else:
                        wait_text = st.empty()
                        wait_text.text("Wait for the image to upload...")
                        upload(new_bird_name, image)
                        wait_text.empty()
                        st.write(
                            "Fininshed upload! Thank you for contributing.")


elif selected == "Home":
    st.title("West bengal bird species classification project")
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.write("Various different bird species are found in west bengal during various times of the year. We have employeed various deep learning technologies such as Transfer learning using GoogLeNet, dataset augmentation and Convolutional Neural Networks to classifiy with accuracy the various birds from limited data.")

    with col2:
        cnn_img = Image.open('CNN.png')
        st.image(cnn_img, caption='Pictorial represenation of CNN model',
                 use_column_width=True)

    st.write("Below are given the training and validation curves for the entire GoogLeNet training phase on the augmented birds dataset.")

    col3, col4 = st.columns([0.5, 0.5])
    with col3:
        training_curve = Image.open('Training_curve.png')
        st.image(training_curve, caption='Training curve for 50 epochs',
                 use_column_width=True)

    with col4:
        validation_curve = Image.open('Validation_curve.png')
        st.image(validation_curve, caption='Validation curve for 50 epochs',
                 use_column_width=True)
