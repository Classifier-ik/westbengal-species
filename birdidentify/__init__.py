from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from flask_session import Session  # helps to store session at back-end
# from flask_login import LoginManager # help with login
from flask_migrate import Migrate  # help to migrate db
from flask_bcrypt import Bcrypt
from flask_mail import Mail
from dotenv import load_dotenv
from pathlib import Path
import os
import sqlite3
from sqlite3 import Connection

db = SQLAlchemy()

app = Flask(__name__)
# Configure session to use filesystem
app.config["SESSION_TYPE"] = "filesystem"

# Make session cookie sid to be signed
app.config['SESSION_USE_SIGNER'] = True

# Cookies older than 15min is rejected
app.config['PERMANENT_SESSION_LIFETIME'] = 15 * 60


# db url from env
dotenv_path = Path('/home/supersub/.env')
load_dotenv(dotenv_path=dotenv_path)


# Sqlite database uri
basedir = os.path.abspath(os.path.dirname(__file__))

def folder_create(path):
    if os.path.exists(path):
        return True
    else:
        os.mkdir(path)
        return True


folder_create(os.path.join(basedir, "static", "tempdir"))
UPLOAD_FOLDER = os.path.join(basedir, "static", "tempdir")
URI_SQLITE_DB = os.path.join(basedir, 'test.db')

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
init_db(conn)
conn.close()

app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///' + os.path.join(basedir, 'test.db') # os.getenv("DB_URL")
# for k, v in os.environ.items():
#     print(f'{k}={v}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# This values need to be stored in environment and called
# Using dummy values for now
# Configuration of a Gmail account for sending mails

app.config['MAIL_SERVER'] = 'smtp.exampleservr.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True


if not os.getenv("MAIL_USERNAME"):
    raise RuntimeError("MAIL_USERNAME is not set")

app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")

if not os.getenv("MAIL_PASSWORD"):
    raise RuntimeError("MAIL_PASSWORD is not set")

app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
app.config['ADMINS'] = [os.getenv("MAIL_USERNAME")]

confirm_token_salt = 'confirm-your-mail-1'

reset_token_salt = 'reset-your-password-2'
mail = Mail(app)

# initialize SQLAlchemy db
db.init_app(app)

# flask-migrate to create automated migrations
migrate = Migrate(app, db)

# Get key from env when in production mode.
if os.getenv("FLASK_ENV") and os.getenv("FLASK_ENV") == "development":
    app.config["SECRET_KEY"] = "OCML3BRawWEUkdbvkduP"
    # app.secret_key = "OCML3BRawWEUeaxcuKH"
    # print("key set")
elif os.getenv("SECRET_KEY"):
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
    # app.secret_key = os.getenv("SECRET_KEY")
    print("got key from env")
else:
    raise RuntimeError("Please set a SECRET_KEY in production mode")

# Get key from env when in production mode.
if os.getenv("FLASK_ENV") and os.getenv("FLASK_ENV") == "development":
    app.config["API_KEY"] = "dummykey"
    # app.secret_key = "SPi35#458fvrsji"
    # print("key set")
elif os.getenv("API_KEY"):
    app.config["API_KEY"] = os.getenv("API_KEY")
    # app.secret_key = os.getenv("API_KEY")
    print("got key from env")
else:
    raise RuntimeError("Please set a API_KEY in production mode")


bcrypt = Bcrypt(app)

# Session will be stored in back-end, faster and longer data can be stored
Session(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


import birdidentify.views  # noqa: E402, F401

if __name__ == "__main__":
    app.run(host='0.0.0.0')
# app.run(host='127.0.0.1', port=5000, debug=True)