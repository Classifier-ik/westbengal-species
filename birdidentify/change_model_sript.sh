python create_csv_train_new.python
rm static/tempdir/model/Googlenet_50_epochs
cp static/tempdir/model/model_transfer.pt static/tempdir/model/Googlenet_50_epochs
rm static/tempdir/model/model_transfer.pt
cp static/tempdir/model/classlabels.pkl static/tempdir/model/classlabels0.pkl
rm static/tempdir/model/classlabels.pkl
cp static/tempdir/model/classlabels1.pkl static/tempdir/model/classlabels.pkl
rm static/tempdir/model/classlabels1.pkl
sudo systemctl restart birdidentify