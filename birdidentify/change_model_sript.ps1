python create_csv_train_new.py
Remove-Item \static\model\model_bird.json
Remove-Item \static\model\model_bird.h5
Copy-Item 'static\model\model_bird1.json' 'static\model\model_bird.json'
Copy-Item 'static\model\model_bird1.h5' 'static\model\model_bird.h5'
Remove-Item \static\model\model_bird1.json
Remove-Item \static\model\model_bird1.h5
Copy-Item 'static\model\classlabels.pkl' 'static\model\classlabels0.pkl'
Remove-Item static\tempdir\model\classlabels.pkl
Copy-Item 'static\model\classlabels1.pkl' 'static\model\classlabels.pkl'
Remove-Item static\model\classlabels1.pkl