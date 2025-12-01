## Environment

This project is developed and tested with `pytorch=2.8`. We recommend using
[conda](https://docs.conda.io/en/latest/)
to ensure all dependencies are installed correctly.
To set up project environment,

```bash
# clone the repository
git clone git@github.com:tianrui-qi/SVF-NF.git
cd SVF-NF
# create the conda environment
conda env create -f environment.yml
conda activate svf-nf
```

## Data

Example data is avaiable at [OSF](https://osf.io/4a5ws/).
To download all data,

```bash
osf -p 4a5ws clone data/
```

Then, rearrange data folder structure as follows:

```bash
mkdir data/frame
mv data/osfstorage/data_all/*.tif data/frame/
mv data/osfstorage/data_all/ExpPSF_605.mat data/psf.mat
rm -r data/osfstorage
```
