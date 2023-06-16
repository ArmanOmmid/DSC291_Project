# DSC 291 Project

# Repository Link
https://github.com/ArmanOmmid/DSC291_Project
# Google Collab Link
https://colab.research.google.com/drive/1BTFDoX7GfhnSCWScxU2gAwKLt57ZPbFn#scrollTo=JdSCvKeJhDHp

## Quickstart Guide

### Running Experiments 
Our main program takes 5 main arguments; other arguments can be observed in **root/src/run_experiment**
1. *config_name* : The config yaml file that configures this experiment; referenced under **root/configs**
2. -D *data_path* : The location to download and/or locate the chosen dataset
3. -E *exp_path* : The path to a folder to store all experiment results
4. -N *exp_name* : The name of the experiment identifying this experiment
5. --download : Flag to download the dataset if it doesn't exist at the given location 

Template:

    python3 repository/src/main.py config_name -D data_path -E exp_path -N exp_name --download

Example:

    python3 repository/src/main.py dsc_complete -D dataset -E experiments -N example --download

Zip all the run results for keeping with:

    zip -r experiments.zip experiments

### Configuring Experiments
Refer to, modify, and create config yaml files under **root/configs** and specify the choice in the *config_name* argument.
In these yaml files, you can choose the model architecture, configurate the architecture, choose the dataset, set the hyperparameters, etc..

### Plotting Results
1. Locate the Jupyter Notebook at **root/src/plotter.ipynb**.
2. Specify the absolute path of your experiments folder containing all the needed experiments by assigning it as a string to the **experiments_path** variable in cell [2].
3. Optional: Include a **styles** variable to specify which experiments are being plotted and how they are styled. 
4. Run the Notebook and observe the plots.