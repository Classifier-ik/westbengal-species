import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch
import pandas
import pickle
from streamlit_option_menu import option_menu

with open('classlabels.pkl', 'rb') as f:
    class_names = pickle.load(f)


def predict(image_path):
    model = torch.load('Googlenet_50_epochs',
                       map_location=torch.device('cpu'))

    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    model.eval()
    outputs = model(batch_t)
    _, predicted = torch.max(outputs, 1)
    title = [class_names[x] for x in predicted]
    prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    classes = pandas.read_csv('bird_dataset.csv', header=None)
    # print("prob: ", float(max(prob)))
    # print("title: ", title[0])
    # print("name: ", classes[0][int(title[0])-1].split('.')[0])
    return (float(max(prob)), classes[0][int(title[0])-1].split('.')[0])


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
    st.header("Upload an image")
    file_up = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

    if file_up is not None:
        image = Image.open(file_up)

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")

        with col2:
            st.write("Your results are served here...")
            score, bird_name = predict(file_up)
            # st.write(results)
            if score > 60:
                st.write("Prediction (name): ",
                         bird_name, ",   \nScore: ", score)
            else:
                st.write("No such bird in database!")

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
