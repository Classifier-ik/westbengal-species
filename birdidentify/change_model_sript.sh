python create_csv_train_new.python
rm static/model/model_bird.json
rm static/model/model_bird.h5
cp static/model/model_bird1.json static/model/model_bird.json
cp static/model/model_bird1.h5 static/model/model_bird.h5
rm static/model/model_bird1.h5
rm static/model/model_bird1.json
cp static/model/classlabels.pkl static/model/classlabels0.pkl
rm static/model/classlabels.pkl
cp static/model/classlabels1.pkl static/model/classlabels.pkl
rm static/model/classlabels1.pkl
sudo systemctl restart birdidentify