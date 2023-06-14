https://gist.github.com/

`model` file contains the model architecture 
`notebooks` file contains the files required to classify and downscale GEFSv12 reanalysis data, and preprocessing of GEFS ensemble input
`postprocessed_data` contains all the GEFS ensemble (input), y_class (ground truth) and y_hr (ground truth) files

Step 1:
Obtain y_hr_train.npy, y_hr_val.npy, and y_hr_test.npy by running 'downscaling_GEFS_reanalysis.ipynb' in 'notebooks' folder

Step 2:
Run model folder with all the training and validating postprocessed .npy files

Step 3:
Obtain the best .h5 models to test on the test .npy files




