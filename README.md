# Solar power generation predictoin

This notebook is an example of creating model for https://github.com/ggear/asystem.

This notebook is asuuming Python 3.6.1.

## Preparation

First, you should get input data.

```sh
$ mkdir -P data
$ wget https://raw.githubusercontent.com/ggear/asystem/master/asystem-amodel/src/test/resources/data/amodel/energy/pristine/training/canonical/txt/csv/none/amodel/1000/engery.csv -O data/energy.csv
$ virtualenv -p python3 venv
$ source venv/bin/activate
(venv)$ pip install -r requirements.txt -c constraints.txt
```

Then, run Power_generation.ipynb with Jupyter notebook.

```sh
(venv)$ jupyter notebook
```
