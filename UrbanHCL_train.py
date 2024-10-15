import argparse
import math

from utils.pre_processing import *
from utils.training import *
from utils.models import *
from utils.properties import *

parser = argparse.ArgumentParser(description='OSM2Vec')
parser.add_argument("-c", type=str, required=True, help='City')

hp = parser.parse_args()
filter_w()

check_data_exists(hp.c)

device = config.device

if not os.path.isdir(hp.c + '/Model'):
    os.mkdir(hp.c + '/Model')

ud = UrbanData(hp.c)
voc = build_category_vocab(ud.entities)
n_polylines = len(set(ud.data.keys()))
model = SSModel(hp.c, config.lm, 768)
train_SS_im(ud, model)
model = read_model(hp.c)
train_SS_svi(ud, model)
model = read_model(hp.c)
train_SS_poi(ud, model)
model = read_model(hp.c)
train_SS_sate(ud, model)
model = read_model(hp.c)
train_SS_region(ud, model)
