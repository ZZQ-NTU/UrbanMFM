# UrbanHCL

### This is the code for 'UrbanHCL' paper

&rarr; Satellite imagery (satellite.tif) is downloaded from Bing Map.

&rarr; Street view images of Singapore and New York Datasets are obtained by Google Map API, and Beijing Dataset is captured by Baidu Map API. The specific format is to shoot street views in the east, south, west and north directions at uniformly produced sampling points.


### Downstream tasks

```
python UrbanHCL_downstream.py -c "city" -t "task"
```

Meaning of the flags and possible values:
* ``-c`` (city): Specify the city you wish to use. Possible values are ``Singapore``, ``"New York"``, ``Beijing``.
* ``-t`` (task): Specify the downstream task. Possible values are ``build_func`` (Singapore), ``pop_density`` (Singapore, NYC and Beijing).

Please note that the first time a task in a city is run, it may require some time to pre-compute the embeddings (using the pre-trained model). After that, the training will always be very fast.

### Pre-processing

If you wish to do it for a city, e.g., Singapore, please use:
```
python UrbanHCL_preprocess.py -c "Singapore"
```

This process will query OSM Overpass API, download and pre-process the data for the requested city. Please be patient, the pre-processing may take several hours, the progress percentage will be shown on the terminal. If you wish to (re-)download the data for an existing city, please delete the city folder before running the command above.


### Pre-training

If you wish to pre-train model of a city, use:

```
python UrbanHCL_train.py -c "city"
```
