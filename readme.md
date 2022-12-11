# westbengal-species

1. Streamlit option

This provides a minimalist option to run the app and check it.
Open the repo and open terminal at that location and run the below commands
```
python3 -m pip install -r requirements.txt
streamlit run app.py
```


2. Flask option

This provides the option to run a full fledged webapp with signup and login. Also it has shellscript option to update the model whenever you wish to.
Open the repo and open terminal at that location and run the below commands to run the app in developement mode(in linux)
```
python3 -m pip install -r requirements1.txt
export DB_URL=sqlite:///home/user/demodb.sqlite3
export FLASK_APP=birdidentify
export FLASK_ENV=development
export MAIL_USERNAME=dummy
export MAIL_PASSWORD=dummy
export API_KEY=dummy_key
flask run
```
Open the repo and open terminal at that location and run the below commands to run the app in developement mode(in Windows)
```
python -m pip install -r requirements1.txt
set DB_URL=sqlite:///home/user/demodb.sqlite3
set FLASK_APP=birdidentify
set FLASK_ENV=development
set MAIL_USERNAME=dummy
set MAIL_PASSWORD=dummy
set API_KEY=dummy_key
flask run
```
Open the repo and open terminal at that location and follow the below steps to run the app in production mode with(nginx and gunicorn in linux)

install the the virtualenv package
```
sudo apt install python3-venv
```

Create a virtualenv
```
python3 -m venv westbengal-speciesenv
```

then activate it with below command and install all the requirements
```
source westbengal-speciesenv/bin/activate
python -m pip install -r requirements.txt
```

Create a service with name birdidentify
```
sudo vim /etc/systemd/system/birdidentify.service
```

with the below content where mtech is the username
```
[Unit]
Description=Gunicorn instance to serve myproject
After=network.target

[Service]
User=mtech
Group=www-data
WorkingDirectory=/home/mtech/westbengal-species
Environment="PATH=/home/mtech/westbengal-species/westbengal-speciesenv/bin"
Environment="SECRET_KEY=bhjvyucjcvyi*2-dsv"
Environment="MAIL_USERNAME=user@example.com"
Environment="MAIL_PASSWORD=dummy"
Environment="DB_URL=sqlite:///home/mtech/westbengal-species/testdb.sqlite3"
Environment="DB_PASSWORD=dummy"
Environment="API_KEY=dummykey"
ExecStart=/home/mtech/westbengal-species/westbengal-speciesenv/bin/gunicorn --workers 3 --bind unix:bird.sock -m 007 wsgi:app

[Install]
WantedBy=multi-user.target
```

now we create a nginx sites-available to localhost
Create a file with name of localhost
```
sudo vim /etc/nginx/sites-available/localhost
```
with the below content
```
server {
    listen 80;
    server_name localhost;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/mtech/westbengal-species/bird.sock;
    }
}
```
next create a symlink
```
sudo ln -s /etc/nginx/sites-available/localhost /etc/nginx/sites-enabled
```