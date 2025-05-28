# UrbanMFM

### This is the code for 'UrbanMFM' paper

&rarr; Satellite imagery (satellite.tif) is downloaded from Bing Map.

&rarr; Street view images of Singapore and New York Datasets are obtained by Google Map API, and Beijing Dataset is captured by Baidu Map API. The specific format is to shoot street views in the east, south, west and north directions at uniformly produced sampling points.


### Pre-processing

If you wish to do it for a city, e.g., Singapore, please use:
```
python UrbanMFM_preprocess.py -c "Singapore"
```

This process will query OSM Overpass API, download and pre-process the data for the requested city. Please be patient, the pre-processing may take several hours, the progress percentage will be shown on the terminal. If you wish to (re-)download the data for an existing city, please delete the city folder before running the command above.


### Pre-training

If you wish to pre-train model of a city, use:

```
python UrbanMFM_train.py -c "city"
```
