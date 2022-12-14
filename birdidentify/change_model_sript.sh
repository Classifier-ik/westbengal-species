python create_csv_train_new.python
rm static/model/Googlenet_50_epochs
cp static/model/model_transfer.pt static/model/Googlenet_50_epochs
rm static/model/model_transfer.pt
cp static/model/classlabels.pkl static/model/classlabels0.pkl
rm static/model/classlabels.pkl
cp static/model/classlabels1.pkl classlabels.pkl
rm static/model/classlabels1.pkl
sudo systemctl restart birdidentify