# Setting up environments for precip rediagnosis scipts/notebooks

The environments for running precip rediagnosis code, either in scripts or in notebooks, depending on the platform and type of work you are doing, are all primarily provided by conda environments.

To create an environment, run the following command
```
conda env create --file requirement.yml
```

where you replace `requirements.yml` with the particular requirements file in the folder required for the part of PRD you want to run. There sveevral different environments as different packages are required for different parts of the data preparation and machine learning pipelines, some of which are mutually incompatible or make for slow/difficult to resolve installations, so we have tried to keep each environment as simple as possible to improve performance.


## Use on Met Office systems

### Use on VDI

To use on VDI, just create the conda environment as above, activate and then either run the script required, or start a jupyter lab server.

### SPICE

To use a script on spice, you can simply actibvate the environment inside your SPICE script. To run the notebooks on spice, use of the Met Office Jupyter Hub installation is advised. Any conda environments you have installed in your homespace wll automaticvally become available as kernels in JupyterHub.

## Use on AzureML

For this project a lot of the wor is being done using the AzureML machine learning platform provided by Microsoft. 

### Notebook server (Compute Instance)

Once you have started a compute instance on AzureML, access it through the Jupter Lab interface. Open a new terminal and once you have clopned the repo, then you can install a conda environemt in the usual way. 
To make available in a notebook, you will need to run an extra command to make your conda environment available as a kernel as follows:

```
conda activate myenv
python -m ipykernel install --user --name=myenv_display_name
```

[source](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084)

### Compute Cluster

To make your environment available on a compute cluster (e.g. fopr training on a GPU), the recommnded approach is to use AzureML Environment assets, which essentially create a docker container based on a requirements file, which is then where the code runs on the compute cluster.

https://azure.github.io/azureml-cheatsheets/docs/cheatsheets/python/v1/environment/ 

This can specified as an argument to the config class or pipeline used to run your script on the cluster:
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines



