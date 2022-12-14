from flask import Flask, redirect, render_template, session, url_for, request, abort, flash, jsonify
import json, os, sqlalchemy
from uuid import uuid4
from datetime import datetime, date
from birdidentify import app, db, confirm_token_salt, reset_token_salt
from .models import (
    User, Test
)
from .custom_decorators import login_required
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
from werkzeug.utils import secure_filename


from itsdangerous import URLSafeTimedSerializer
from . import bcrypt
# from flask_login import login_user, login_required, current_user
from sqlalchemy import exc  # , func

# Serializer for generating random tokens
ts = URLSafeTimedSerializer(app.config['SECRET_KEY'])


with open(os.path.join(app.config['UPLOAD_FOLDER'],'model', 'classlabels.pkl'), 'rb') as f:
    class_names = pickle.load(f)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']


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


def predictly(image_path):
    model = torch.load(os.path.join(app.config['UPLOAD_FOLDER'],'model', 'Googlenet_50_epochs'),
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
    # title = [class_names[x] for x in predicted]
    prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    # classes = pandas.read_csv('bird_dataset.csv', header=None)
    # print("prob: ", float(max(prob)))
    # print("title: ", title[0])
    # print("name: ", classes[0][int(title[0])-1].split('.')[0])
    return (float(max(prob)), class_names[predicted])


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='tempdir/' + filename), code=301)


@app.route('/viewrecord/', methods=['GET', 'POST'])
def viewrec():
    """
    View prediction 
    """
    if request.method == 'GET':
        if session.get('logged_in'):
            user = User.query.filter_by(username=session['email']).first()
            if not user.isAdmin:
                testimlist = Test.query.filter_by(user_id = user.id, validity= 0).order_by(Test.validity).all()
            else :
                testimlist = Test.query.order_by(Test.validity).all()
            
            return render_template("viewrecord.html", testimlist=testimlist)
        else:
            return redirect(url_for('login'))


@app.route('/update/<int:key>/', methods=['GET', 'POST'])
@login_required(['admin'])
def update(key):
    """Update selected subject"""
    if request.method == 'GET':
        test = Test.query.filter_by(id=key).one_or_none()
        return render_template("update.html", entry=test)
    if request.method == 'POST':
        """
        Provide a method for adding new subjects
        """
        idy = request.values.get('id')  # Image id
        userinput = request.values.get('userinput')  # User input
        validity = request.values.get('validity')  # Validity

        # try:
        # create new subject.
        get_img = Test.query.filter_by(id=idy).one_or_none()
        get_img.userinput = userinput
        get_img.validity = validity

        # add the new user to the database
        Test.query.filter_by(id=idy).update(
            {"userinput": userinput, "validity": validity}
        )
        db.session.commit()

        flash(f'Please note down image with id = {idy} updated with validity = {validity}', 'danger')

        return redirect(url_for('update', key=idy))


@app.route('/upload/',methods=['GET','POST'])
def upload():
    #the upload button page
    return render_template("upload.html")


@app.route('/predict/',methods=['GET','POST'])
def predict():
    #the prediction page
    if request.method == 'POST':
        f = request.files['file']
        if f.filename=='':
            flash('No file part')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            rand_str = str(uuid.uuid4())
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], rand_str + filename))
            path = rand_str + filename
            score, bird_name = predictly(os.path.join(app.config['UPLOAD_FOLDER'], rand_str + filename))

            return render_template('predict.html', title='Success', score=score, bird_name=bird_name, user_image=path)
        return redirect(url_for('upload'))
    return redirect(url_for('dash'))


@app.route('/userinput/',methods=['GET','POST'])
def userinput():
    #the prediction page
    if request.method == 'POST':
        filepath = request.values.get('filepath')  # Your form's
        predicted = request.values.get('predicted')  # Your form's
        userinput = request.values.get('userinput')  # Your form's
        validity = request.values.get('validity')  # Your form's
        user = User.query.filter_by(username=session.get('email')).one_or_none()
        print(validity)
        if user:
            img_data = Test(
                                filepath=filepath,
                                predicted=predicted,
                                userinput=userinput,
                                validity=int(validity),
                                user_id=user.id
                            )
            # add the new user to the database
            db.session.add(img_data)
            db.session.commit()
            return redirect(url_for('update',key = user.id))
        else:
            img_data = Test(
                                filepath=filepath,
                                predicted=predicted,
                                userinput=userinput,
                                validity=int(validity),
                            )
            # add the new user to the database
            db.session.add(img_data)
            db.session.commit()
        return render_template('userinput.html', title='Success', bird_name=predicted, validity=validity, user_image=filepath) 
    return redirect(url_for('upload'))


@app.route('/')
def index():
    """
    Home Page
    don't know what's needed or not needed
    """
    if session.get('category') == "admin":
        return redirect(url_for('dash'))
    elif session.get('category') == "user":
        return redirect(url_for('dash'))
    else:
        return render_template("home-page-edit.html")


@app.route('/logout', methods=['GET'])
def logout():
    if session.get('logged_in'):
        session['logged_in'] = False
        session.pop('email')
        session.pop('category')
    flash('Logged out', 'danger')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.values.get('email')  # Your form's
        password = request.values.get('password')  # input names

        # let's start with logging out any existing user.
        session['logged_in'] = False  # enough

        if '@' in username:
            user = User.query.filter_by(username=username.lower()).first()

            # check if user actually exists
            # take the user supplied password, hash it,
            # and compare it to the hashed password in db
            if not user or not bcrypt.check_password_hash(user.password, password):
                # if user doesn't exist or password is wrong, reload the page
                flash('Please check your login details and try again.', 'danger')
                return redirect(url_for('login'))
            # if the above check passes, then we know the user has the right credentials
            session['email'] = str(user.username)
            if user.isAdmin:
                session['category'] = 'admin'
                session['logged_in'] = True
                flash("Logged in successfully", 'success')
                return redirect(url_for('dash'))
            else:
                session['category'] = 'user'
                session['logged_in'] = True
                flash("Logged in successfully", 'success')
                return redirect(url_for('dash'))
        else:
            flash("No login info provided", 'danger')
            return redirect(url_for('login'))
    else:
        return render_template("login.html")


@app.route('/dashboard/profile/')
def profile():
    """
    Display profile info
    """
    category = session['category']
    if category == 'admin':
        user = User.query.filter_by(username=session['email']).first()
    elif category == 'user':
        user = User.query.filter_by(username=session['email']).first()
    return render_template('profile.html', user=user)


@app.route('/dashboard/')
@login_required(['admin','user'])
def dash():
    """
    dashboard
    """
    user = User.query.filter_by(username=session['email']).first()
    return render_template('dash.html', user=user)


@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        """
        Provide a method for registering new users.
        Admin is made using direct access to db
        """
        password = request.values.get('password')  # input names
        email = request.values.get('email')
        if email:
            email = email.lower()

        is_admin = request.values.get('category') # remove after creating one admin
        
        # create new user with the form data. Hash the password so plaintext version isn't saved.
        try:
            new_user = User(
                username=email,
                isAdmin=is_admin,
                password=bcrypt.generate_password_hash(password)
            )
            # add the new user to the database
            db.session.add(new_user)
            db.session.commit()
        except sqlalchemy.exc.IntegrityError:
            flash('Email already exists in records ' + email + ' . Contact system admin.', 'danger')
            return redirect(request.url)

        # mail_sent = send_confirm_email(new_user)
        # Send back to the home page
        flash("User with name " + new_user.username
              + " is successfully registered in the system",
              'success')
        '''
        if mail_sent:
            flash('Please check ' + new_user.email + ' for confirmation link', 'success')
        else:
            flash('Unable to send email to ' + new_user.email + ' . Contact system admin', 'danger')
        '''
        return redirect(request.url)
    else:
        return render_template("register.html")