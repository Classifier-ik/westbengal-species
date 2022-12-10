import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch
import os
import pandas
import sqlite3
from sqlite3 import Connection
import pickle
import hashlib
import uuid
from streamlit_option_menu import option_menu
from st_aggrid import GridOptionsBuilder, AgGrid
from st_aggrid.shared import GridUpdateMode, DataReturnMode

with open('classlabels.pkl', 'rb') as f:
    class_names = pickle.load(f)


def folder_create(path):
    if os.path.exists(path):
        return True
    else:
        os.mkdir(path)
        return True


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


# Sqlite database uri
URI_SQLITE_DB = "test.db"

def init_db(conn: Connection):
    conn.execute("""CREATE TABLE IF NOT EXISTS test(
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filepath TEXT NOT NULL UNIQUE,
                            predicted TEXT NOT NULL,
                            userinput TEXT NOT NULL,
                            user_id INTEGER,
                            validity INTEGER
                    );""")
    conn.execute("""CREATE TABLE IF NOT EXISTS userstable
                        (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT NOT NULL UNIQUE,
                            password TEXT NOT NULL,
                            isadmin INTEGER NOT NULL DEFAULT 0
                        );
                    """)
    conn.commit()
    

st.set_page_config(layout="wide")
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(hash_funcs={Connection: id})
def get_connection(path: str):
    """Put the connection in cache to reuse if path does not change between Streamlit reruns.
    NB : https://stackoverflow.com/questions/48218065/programmingerror-sqlite-objects-created-in-a-thread-can-only-be-used-in-that-sa
    """
    return sqlite3.connect(path, check_same_thread=False)


conn = get_connection(URI_SQLITE_DB)
init_db(conn)
c = conn.cursor()

curr_path = os.path.realpath(os.path.dirname(__file__))
folder_create(os.path.join(curr_path,"tempdir"))

def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()


def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchone()
    return data


def view_all_images():
    c.execute('SELECT * FROM test')
    data = c.fetchall()
    return data


def view_my_images(user_id):
    c.execute('SELECT * FROM test WHERE user_id =?',(user_id,))
    data = c.fetchall()
    return data


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


selected = option_menu(
    menu_title=None,
    options=["Home", "Project","Login","SignUp"],
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
        file_details = {"FileName":file_up.name,"FileType":file_up.type}
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            # insert checking for file name repeation in single user and within various user
            image.save(os.path.join(curr_path,"tempdir",str(uuid.uuid4())+file_up.name))
            
        with col2:
            st.write("Your results are served here...")
            score, bird_name = predict(file_up)
            # st.write(results)
            if score > 60:
                st.write("Prediction (name): ",
                        bird_name, ",   \nScore: ", score)
                user_val = st.text_area('Confirm bird name', '''Enter bird name''')
            else:
                st.write("No such bird in database!")
                user_val = st.text_area('Confirm bird name', '''Enter bird name''')
        try:
            c.execute('INSERT INTO test(filepath,predicted,userinput,validity) VALUES (?,?,?,?)',(os.path.join(curr_path,"tempdir",file_up.name),bird_name,user_val,0))
        except:
            c.execute('UPDATE test SET userinput = ? where filepath=?',(user_val, os.path.join(curr_path,"tempdir",file_up.name)))
        conn.commit()

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


elif selected == "Login":
        st.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:

                st.success("Logged In as {}".format(username))

                task = st.selectbox("Task",["View uploaded image status","Predict"])
                if task == "View uploaded image status":
                    st.subheader("View images")
                    print(result)
                    if result[3] == 1:
                        user_result = view_all_images()
                        clean_db = pandas.DataFrame(user_result,columns=["id","filepath","predicted","userinput","user_id","validity"])
                        gb = GridOptionsBuilder.from_dataframe(clean_db)
                        gb.configure_column("id", header_name=("id"), editable=False)
                        gb.configure_column("filepath", header_name=("filepath"), editable=False)
                        gb.configure_column("predicted", header_name=("predicted"), editable=False)
                        gb.configure_column("userinput", header_name=("userinput"), editable=True)
                        gb.configure_column("user_id", header_name=("user_id"), editable=False)
                        gb.configure_column("validity", header_name=("validity"), editable=True,precision=0)
                    else:
                        user_result = view_my_images(result[0])
                        clean_db = pandas.DataFrame(user_result,columns=["id","filepath","predicted","userinput","user_id","validity"])
                        gb = GridOptionsBuilder.from_dataframe(clean_db)
                        gb.configure_column("id", header_name=("id"), editable=False)
                        gb.configure_column("filepath", header_name=("filepath"), editable=False)
                        gb.configure_column("predicted", header_name=("predicted"), editable=False)
                        gb.configure_column("userinput", header_name=("userinput"), editable=True)
                        gb.configure_column("user_id", header_name=("user_id"), editable=False)
                        gb.configure_column("validity", header_name=("validity"), editable=False,precision=0)

                    gridOptions = gb.build()
                    dta = AgGrid(clean_db,
                                    gridOptions=gridOptions,
                                    reload_data=False,
                                    height=200,
                                    editable=True,
                                    theme="streamlit",
                                    data_return_mode=DataReturnMode.AS_INPUT,
                                    update_mode=GridUpdateMode.VALUE_CHANGED)
                    if st.button("Update"):
                        for i in range(len(dta['data'])): # or you can use for i in range(tdf.shape[0]):
                            # st.caption(f"df line: {clean_db.loc[i][0]} | {clean_db.loc[i][1]} || AgGrid line: {dta['data']['Name'][i]} | {dta['data']['Amt'][i]}")
                                print(clean_db.loc[i]['validity'], dta['data']['validity'][i], clean_db.loc[i]['userinput'], dta['data']['userinput'][i])
                                # check if any change has been done to any cell in any col by writing a caption out
                                if clean_db.loc[i]['validity'] != dta['data']['validity'][i]:
                                    print(int(dta['data']['validity'][i]), clean_db.loc[i]['id'])
                                    if int(dta['data']['validity'][i]):
                                        c.execute('''UPDATE test SET validity=? where id=?''',(1, clean_db.loc[i]['id']))
                                    else:
                                        c.execute('''UPDATE test SET validity=? where id=?''',(0, clean_db.loc[i]['id']))
                                    conn.commit()
                                    print("Validity change Commited")
                                    stringy = view_all_images()
                                    print(stringy)
                                    # st.caption(f"Name column data changed from {tdf.loc[i]['Name']} to {dta['data']['Name'][i]}...")
                                    # consequently, you can write changes to a database if/as required

                                if clean_db.loc[i]['userinput'] != dta['data']['userinput'][i] and result[3]==1:
                                    c.execute('''UPDATE test SET userinput=? where id=?''',(dta['data']['userinput'][i], clean_db.loc[i]['id']))
                                    conn.commit()
                                    print("user input change Commited")
                                    # st.caption(f"Amt column data changed from {tdf.loc[i]['Amt']} to {dta['data']['Amt'][i]}...")
                        clean_db = dta['data']    # overwrite df with revised aggrid data; complete dataset at one go
                        # tdf.to_csv(vpth + 'file1.csv', index=False)  # re/write changed data to CSV if/as requir
                        st.dataframe(clean_db)
                elif task == "Predict":
                    st.header("Upload an image \U0001F447\U0001F447\U0001F447")
                    file_up = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

                    if file_up is not None:
                        image = Image.open(file_up)
                        file_details = {"FileName":file_up.name,"FileType":file_up.type}
                        col1, col2 = st.columns([0.5, 0.5])
                        with col1:
                            st.image(image, caption='Uploaded Image.', use_column_width=True)
                            st.write("")
                            # insert checking for file name repeation in single user and within various user
                            image.save(os.path.join(curr_path,"tempdir",str(result[0])+"_"+file_up.name))
                            '''  
                            with open(os.path.join(curr_path,"tempdir",file_up.name), 'wb') as handler:
                                handler.write(image)
                            '''

                        with col2:
                            st.write("Your results are served here...")
                            score, bird_name = predict(file_up)
                            # st.write(results)
                            if score > 60:
                                st.write("Prediction (name): ",
                                        bird_name, ",   \nScore: ", score)
                                user_val = st.text_area('Confirm bird name', '''Enter bird name''')
                            else:
                                st.write("No such bird in database!")
                                user_val = st.text_area('Confirm bird name', '''Enter bird name''')
                        try:
                            c.execute('INSERT INTO test(filepath,predicted,userinput,user_id,validity) VALUES (?,?,?,?,?)',(os.path.join(curr_path,"tempdir",file_up.name),bird_name,user_val,result[0],0))
                        except:
                            c.execute('UPDATE test SET userinput = ? where filepath=? and user_id=?',(user_val, os.path.join(curr_path,"tempdir",file_up.name), result[0]))
                        conn.commit()
            else:
                st.warning("Incorrect Username/Password")


elif selected == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')

        if st.button("Signup"):
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
