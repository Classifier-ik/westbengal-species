def updatemode():
    # Author: Anonymous
    # project: Cat vs Dog

    import sqlite3
    from sqlite3 import Connection
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import cv2
    import matplotlib.pyplot as plt
    import glob
    from tqdm import tqdm
    
    from skimage import io, transform
    from keras.utils import to_categorical
    import time
    from sklearn.model_selection import train_test_split
    import pickle
    seed = 333
    np.random.seed(seed)

    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

    import os
    # os.environ['KERAS_BACKEND'] = 'theano'

    #path to images
    img_dir = "static/train"
    basedir = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(basedir, "static")
    URI_SQLITE_DB = os.path.join(basedir, 'test.db')
    def folder_create(path):
        if os.path.exists(path):
            return True
        else:
            os.mkdir(path)
            return True

    data_dir = URI_SQLITE_DB = os.path.join(basedir, 'static', 'train')
    folder_create(data_dir)
    '''
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


    def get_connection(path: str):
        """Put the connection in cache to reuse if path does not change between Streamlit reruns.
        NB : https://stackoverflow.com/questions/48218065/programmingerror-sqlite-objects-created-in-a-thread-can-only-be-used-in-that-sa
        """
        return sqlite3.connect(path, check_same_thread=False)


    conn = get_connection(URI_SQLITE_DB)
    # init_db(conn)

    df = pd.read_sql_query("SELECT * FROM test", conn)
    file_data = df[['id', 'filepath','validity','userinput']]
    file_data = file_data[file_data['validity'] == 1]
    file_data['userinput'] = file_data['userinput'].str.replace(" ","-")

    for value in file_data.userinput.unique():
        path = os.path.join(basedir, "static", "train", value)
        folder_create(path)

    for index, row in file_data.iterrows():
        shutil.copyfile(os.path.join(basedir, "static", "tempdir", row["filepath"]), os.path.join(basedir, "static", "train", row["userinput"], row["filepath"]))

    # img_dir2 = "../input/horses-or-humans-dataset/horse-or-human/horse-or-human"

    # list all available images type
    # print(os.listdir(img_dir))
    # print(os.listdir(img_dir2))
    '''
    def load_data(img_dir):
        X = []
        y = []
        labels = []
        idx = 0
        for i,folder_name in enumerate(os.listdir(img_dir)):
            labels.append(folder_name)
            for file_name in tqdm(os.listdir(f'{img_dir}/{folder_name}')):
                if file_name.endswith('jpg'):
                    im = cv2.imread(f'{img_dir}/{folder_name}/{file_name}')
                    if im is not None:
                        im = cv2.resize(im, (100, 100))
                        X.append(im)
                        y.append(idx)
                elif file_name.endswith('png'):
                    im = cv2.imread(f'{img_dir}/{folder_name}/{file_name}')
                    if im is not None:
                        im = cv2.resize(im, (100, 100))
                        if len(im.shape) > 2 and im.shape[2] == 4:
                            #convert the image from RGBA2RGB
                            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
                        X.append(im)
                        y.append(idx)
            idx+=1
        X = np.asarray(X)
        y = np.asarray(y)
        labels = np.asarray(labels)
        return X,y,labels


    X,y,labels = load_data(img_dir)

    with open(os.path.join(UPLOAD_FOLDER,'model','classlabels1.pkl'), 'wb') as fh:
        labelo = np.array(labels)
        labelo = labelo.reshape(-1)
        labelo = np.asarray(labelo)
        pickle.dump(labelo, fh)

    #fix y
    y = y.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    train_img = X_train
    train_labels = y_train
    test_img = X_test
    test_labels = y_test
    train_img.shape, train_labels.shape, test_img.shape, test_labels.shape

    #one-hot-encode the labels
    num_classes = len(labels)
    train_labels_cat = to_categorical(train_labels,num_classes)
    test_labels_cat = to_categorical(test_labels,num_classes)
    train_labels_cat.shape, test_labels_cat.shape

    # re-shape the images data
    train_data = train_img
    test_data = test_img
    train_data.shape, test_data.shape

    # shuffle the training dataset & set aside val_perc % of rows as validation data
    for _ in range(5): 
        indexes = np.random.permutation(len(train_data))

    # randomly sorted!
    train_data = train_data[indexes]
    train_labels_cat = train_labels_cat[indexes]

    # now we will set-aside val_perc% of the train_data/labels as cross-validation sets
    val_perc = 0.10
    val_count = int(val_perc * len(train_data))
    print(val_count)

    # first pick validation set
    val_data = train_data[:val_count,:]
    val_labels_cat = train_labels_cat[:val_count,:]

    # leave rest in training set
    train_data2 = train_data[val_count:,:]
    train_labels_cat2 = train_labels_cat[val_count:,:]

    train_data2.shape, train_labels_cat2.shape, val_data.shape, val_labels_cat.shape, test_data.shape, test_labels_cat.shape

    # a utility function that plots the losses and accuracies for training & validation sets across our epochs
    def show_plots(history):
        """ Useful function to view plot of loss values & accuracies across the various epochs """
        loss_vals = history['loss']
        val_loss_vals = history['val_loss']
        epochs = range(1, len(history['acc'])+1)
        
        f, ax = plt.subplots(nrows=1,ncols=2,figsize=(16,4))
        
        # plot losses on ax[0]
        ax[0].plot(epochs, loss_vals, color='navy',marker='o', linestyle=' ', label='Training Loss')
        ax[0].plot(epochs, val_loss_vals, color='firebrick', marker='*', label='Validation Loss')
        ax[0].set_title('Training & Validation Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend(loc='best')
        ax[0].grid(True)
        
        # plot accuracies
        acc_vals = history['acc']
        val_acc_vals = history['val_acc']

        ax[1].plot(epochs, acc_vals, color='navy', marker='o', ls=' ', label='Training Accuracy')
        ax[1].plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Validation Accuracy')
        ax[1].set_title('Training & Validation Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend(loc='best')
        ax[1].grid(True)
        
        plt.show()
        plt.close()
        
        # delete locals from heap before exiting
        del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals


    def print_time_taken(start_time, end_time):
        secs_elapsed = end_time - start_time
        
        SECS_PER_MIN = 60
        SECS_PER_HR  = 60 * SECS_PER_MIN
        
        hrs_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_HR)
        mins_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_MIN)
        
        if hrs_elapsed > 0:
            print('Time taken: %d hrs %d mins %d secs' % (hrs_elapsed, mins_elapsed, secs_elapsed))
        elif mins_elapsed > 0:
            print('Time taken: %d mins %d secs' % (mins_elapsed, secs_elapsed))
        elif secs_elapsed > 1:
            print('Time taken: %d secs' % (secs_elapsed))
        else:
            print('Time taken - less than 1 sec')


    def get_commonname(idx):
        sciname = labels[idx][0]
        return sciname


    from keras.layers import Flatten, Dense, Dropout


    import numpy as np
    from keras.utils.np_utils import to_categorical

    from keras.models import Sequential, model_from_json, Model
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,Activation,MaxPooling2D
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import LearningRateScheduler
    from keras.datasets import mnist
    from keras.models import load_model
    from sklearn.model_selection import train_test_split
    from keras.utils import np_utils
    from PIL import Image


    #data augmentation
    datagen = ImageDataGenerator(
            rotation_range=30,
            zoom_range = 0.25,  
            width_shift_range=0.1, 
            height_shift_range=0.1)
    # datagen = ImageDataGenerator(
    #     rotation_range=8,
    #     shear_range=0.3,
    #     zoom_range = 0.08,
    #     width_shift_range=0.08,
    #     height_shift_range=0.08)


    #create multiple cnn model for ensembling
    #model 1
    '''
    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (100, 100, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))


    model.add(Conv2D(256, kernel_size = 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    # use adam optimizer and categorical cross entropy cost
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # after each epoch decrease learning rate by 0.95
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

    # train
    epochs = 50
    j=0
    start_time = time.time()
    history = model.fit_generator(datagen.flow(train_data2, train_labels_cat2, batch_size=64),epochs = epochs, steps_per_epoch = train_data2.shape[0]/64,validation_data = (val_data, val_labels_cat), callbacks=[annealer], verbose=1)
    end_time = time.time()
    print_time_taken(start_time, end_time)


    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(j+1,epochs,history.history['accuracy'][epochs-1],history.history['val_accuracy'][epochs-1]))
    '''

    # load json and create model
    json_file = open('static/model/model_bird_foreign.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("static/model/model_bird_foreign.h5")
    print("Loaded model from disk")
    print(len(loaded_model.layers))

    for i in range(len(loaded_model.layers)):
        loaded_model.layers[i].trainable = False
        print(loaded_model.layers[i].name)
 
    model = Sequential()
    for i in range(len(loaded_model.layers)):
        if i <= 22:
            model.add(loaded_model.layers[i])

    model.add(Flatten(name="flatten"))
    model.add(Dropout(0.4, name="dropout_3"))
    model.add(Dense(num_classes, activation='softmax'))

    # use adam optimizer and categorical cross entropy cost
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # after each epoch decrease learning rate by 0.95
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
    print("\n new model \n ")
    for i in range(len(model.layers)):
        print(i, model.layers[i].name)
    
    print(len(model.layers))
    # train
    epochs = 20
    j=0
    start_time = time.time()
    history = model.fit_generator(datagen.flow(train_data2, train_labels_cat2, batch_size=64),epochs = epochs, steps_per_epoch = train_data2.shape[0]/64,validation_data = (val_data, val_labels_cat), callbacks=[annealer], verbose=1)
    end_time = time.time()
    print_time_taken(start_time, end_time)


    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(j+1,epochs,history.history['accuracy'][epochs-1],history.history['val_accuracy'][epochs-1]))

    test_loss, test_accuracy = model.evaluate(test_data, test_labels_cat, batch_size=64)
    print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))

    from keras.models import model_from_json
    from keras.models import load_model

    # serialize model to JSON
    #  the keras model which is trained is defined as 'model' in this example
    model_json = model.to_json()


    with open("static/model/model_bird2.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("static/model/model_bird2.h5")


if __name__=="__main__":
    updatemode()
