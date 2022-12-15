python create_csv_train_new.py
Remove-Item \static\tempdir\model\Googlenet_50_epochs
Copy-Item 'static\tempdir\model\model_transfer.pt' 'static\tempdir\model\Googlenet_50_epochs'
Remove-Item static\tempdir\model\model_transfer.pt
Copy-Item 'static\tempdir\model\classlabels.pkl' 'static\tempdir\model\classlabels0.pkl'
# Remove-Item static\tempdir\model\classlabels.pkl
Copy-Item 'static\tempdir\model\classlabels1.pkl' 'static\tempdir\model\classlabels.pkl'
Remove-Item static\tempdir\model\classlabels1.pkl