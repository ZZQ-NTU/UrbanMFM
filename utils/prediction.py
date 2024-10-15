import copy
import csv
import random

import numpy as np
import pandas as pd
import torch

import torch.nn.functional as F
import torchvision.transforms as transforms
import clip
from PIL import Image
import tifffile

from torchmetrics.functional import r2_score
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
from transformers import CLIPProcessor, CLIPModel
import concurrent.futures

from utils.pre_processing import *
from utils.training import *
from utils.models import *


def precompute_embs_pop_d(city, ud, model, pop_d):

    device = config.device

    district_emb = {}
    key = 0
    tok = Tokenizer(config.lm)
    count = 0
    h = 32
    dropout = 0.1
    max_distance = 500.0
    gatedfusion_triple = GatedFusion(model.emb_size, 3).to(device)
    gatedfusion_double = GatedFusion(model.emb_size, 2).to(device)
    model = model.to(device)
    svi_generator = Semantic_Consistency(temperature=0.5).to(device)
    spatial_aware_svi_generator = Spatial_Aware_Semantic_Consistency(temperature=0.5).to(device)

    tif_path = str(city) + '/Data/satellite.tif'

    with tifffile.TiffFile(tif_path) as tif:
        sate_img = tif.pages[0].geotiff_tags
        tif_image = tif.asarray()

    for d in pop_d:

        count += 1

        # if random.choice([True, False]):
        # continue

        district = pop_d[d]

        d_polygon = Polygon([[p[1], p[0]] for p in district['polygon']])
        semantic_nodes = []
        polygons = []

        pe = PE((d_polygon.centroid.x, d_polygon.centroid.y), config.pe_size)

        coordinates_list = []

        for coord in d_polygon.exterior.coords:
            coordinates_list.append((coord[0], coord[1]))

        local_embs = []

        with torch.no_grad():

            for e in ud.entities:

                entity = ud.entities[e]
                entity_p = Point(entity['lat'], entity['lon'])

                if d_polygon.contains(entity_p):
                    poi_node = [e]
                    poi = torch.tensor(serialize_data([ud.entities[k] for k in poi_node], tok, add_zero=True))
                    poi_m = torch.tensor(np.where(poi != 0, 1, 0))
                    poi = poi.to(device)
                    poi_m = poi_m.to(device)
                    poi_emb = model.projection_p(
                        torch.mean(torch.mean(model.lm(poi, attention_mask=poi_m)[0][:, :, :], 0), 0)).to(device)

                    svi_memory = []
                    dist_memory = []
                    dist_list = ud.poi[e]['svi_dist']
                    id_list = ud.poi[e]['svi']
                    angle_list = ud.poi[e]['angle']
                    for i in range(len(id_list)):
                        east_svi_emb, south_svi_emb, west_svi_emb, north_svi_emb = compute_svi_emb(
                            model, city, str(id_list[i]))
                        svi_emb = compute_svi(angle_list[i],
                                              east_svi_emb, south_svi_emb, west_svi_emb, north_svi_emb)

                        svi_memory.append(svi_emb)
                        dist_memory.append(max_distance / (max_distance + dist_list[i]))

                    svi_tensor = torch.stack(svi_memory).to(device)
                    distance_tensor = torch.tensor(dist_memory).to(device)
                    svi_emb = spatial_aware_svi_generator(poi_emb, svi_tensor, distance_tensor)

                    if entity['type'] == 'node':

                        poly_emb = torch.zeros(model.emb_size)

                        poi_emb = poi_emb.unsqueeze(0)
                        svi_emb = svi_emb.unsqueeze(0)
                        poly_emb = poly_emb.unsqueeze(0)

                        poi_feat = poi_emb.to(device)
                        svi_feat = svi_emb.to(device)
                        poly_feat = poly_emb.to(device)

                        local_emb = gatedfusion_double(poi_feat, svi_feat, poly_feat).squeeze()
                        local_embs.append(local_emb)

                        semantic_nodes.append(e)

                    elif entity['type'] == 'polygon':

                        raster = poly_to_raster(Polygon(entity['points']))
                        if not len(raster):
                            continue
                        image = torch.tensor(raster).unsqueeze(0).float().to(device)
                        image_emb = model.projection_img(model.cv(image).squeeze())
                        s = torch.tensor(norm_surface(entity['area'])).unsqueeze(-1).to(device)
                        s_emb = model.projection_s(s)

                        poly_emb = image_emb + s_emb / 2

                        semantic_nodes.append(e)
                        polygons.append(e)

                        poi_emb = poi_emb.unsqueeze(0)
                        svi_emb = svi_emb.unsqueeze(0)
                        poly_emb = poly_emb.unsqueeze(0)

                        poi_feat = poi_emb.to(device)
                        svi_feat = svi_emb.to(device)
                        poly_feat = poly_emb.to(device)

                        local_emb = gatedfusion_triple(poi_feat, svi_feat, poly_feat).squeeze()

                        local_embs.append(local_emb)

            satellite_image = extract_image_from_coords(sate_img, tif_image, coordinates_list)
            if satellite_image.size > 0:
                satellite_image = Image.fromarray(satellite_image)
                if satellite_image.mode != 'RGB':
                    satellite_image = satellite_image.convert('RGB')
                satellite_input_tensor = model.preprocess(satellite_image).to(device)
                satellite_input_batch = satellite_input_tensor.unsqueeze(0)
                global_emb = model.projection_img(model.cv(satellite_input_batch).squeeze())
            else:
                global_emb = torch.zeros(model.emb_size)

        if len(local_embs) > 0:
            local_emb = torch.mean(torch.stack(local_embs), 0)
        else:
            local_emb = torch.zeros(model.emb_size)

        if len(semantic_nodes) == 0:
            continue

        # thousands of people per square kilometer

        district_emb[key] = {'pos': list(pe),'local': local_emb.tolist(), 'global': global_emb.tolist(),
                             'pop_density': district['population'], 'nodes_id': semantic_nodes, 'poly_ids': polygons}

        key += 1

        s = 'Precomputing districts embeddings... (' + str(round(count / len(list(pop_d)) * 100, 2)) + '%)'
        in_place_print(s)

    print(config.flush)
    save(district_emb, str(city) + '/Data/pop_d_embeddings.json')


def population_density(city, ud, model, training=True):
    with open(str(city) + '/Data/pop_density.json', encoding='utf-8') as file:
        pop_d = json.load(file)

    if not os.path.isfile(city + '/Data/pop_d_embeddings.json'):
        precompute_embs_pop_d(city, ud, model, pop_d)

    with open(str(city) + '/Data/pop_d_embeddings.json', encoding='utf-8') as file:
        pop_d_embs = json.load(file)

    keys = list(pop_d_embs.keys())

    n_runs = 10

    rmse, l1, r2, mape_ = [], [], [], []

    for i in range(n_runs):
        random.shuffle(keys)
        train = keys[:len(keys) // 10 * 6]
        valid = keys[len(keys) // 10 * 6:len(keys) // 10 * 8]
        test = keys[len(keys) // 10 * 8:]

        max_p = max([pop_d_embs[h]['pop_density'] for h in train + valid + test])
        # max_p = 1.0

        ds_model = HPRegressor(model.emb_size)
        train_pd_prediction(city, pop_d_embs, ds_model, None, max_p, train, valid, training, 'pd')

        with open(str(city) + '/Model/model_hp.pkl', 'rb') as file:
            ds_model = pickle.load(file)

        this_rmse, this_l1, this_r2 = validate_pd_prediction(pop_d_embs, ds_model, None, config.bs, max_p, test, 0,
                                                             True, 'pd')
        rmse.append(this_rmse)
        l1.append(this_l1)
        r2.append(this_r2)

    print("\nFinal results out of " + str(n_runs) + ' runs:\n')
    print('RMSE: ' + str(round(float(np.mean(np.array(rmse))), 2)) + ' (' + str(
        round(float(np.std(np.array(rmse))), 2)) + ') | MAE: ' + str(
        round(float(np.mean(np.array(l1))), 2)) + ' (' + str(round(float(np.std(np.array(l1))), 2)) + ') | R2: ' + str(
        round(float(np.mean(np.array(r2))), 2)) + ' (' + str(
        round(float(np.std(np.array(r2))), 2)) + ')')


def train_pd_prediction(city, hd_embs, ds_model, hd, max_p, train, valid, training, task):
    if task == 'hp':
        epochs = config.ds_epochs
        lr = config.ds_lr
    else:
        epochs = config.ds_epochs * 2
        lr = config.ds_lr

    opt = optim.Adam(params=ds_model.parameters(), lr=lr)
    bs = config.ds_bs
    criterion = nn.MSELoss()

    best_rmse = math.inf

    for epoch in range(epochs):

        if not training:
            break

        print_epoch(epoch + 1, epochs)

        opt.zero_grad()

        idx = 0
        step = 0

        while idx < len(train):

            ds_model.train()

            if idx > len(train) - bs:
                batch_l = train[idx:]
                if task == 'hp':
                    batch_y = torch.tensor([hd[str(int(h) // 2)]['price'] / max_p for h in batch_l])
                else:
                    batch_y = torch.tensor([hd_embs[h]['pop_density'] / max_p for h in batch_l])

            else:
                batch_l = train[idx:idx + bs]
                if task == 'hp':
                    batch_y = torch.tensor([hd[str(int(h) // 2)]['price'] / max_p for h in batch_l])
                else:
                    batch_y = torch.tensor([hd_embs[h]['pop_density'] / max_p for h in batch_l])

            batch_global = []
            batch_local = []
            batch_p = []

            for element in batch_l:
                batch_global.append(torch.tensor(hd_embs[element]['global']))
                batch_local.append(torch.tensor(hd_embs[element]['local']))
                batch_p.append(torch.tensor([pos / 10 for pos in hd_embs[element]['pos']]))

            batch_global = torch.stack(batch_global)
            batch_local = torch.stack(batch_local)
            batch_p = torch.stack(batch_p)

            p = ds_model(batch_global, batch_local, batch_p)

            loss = torch.sqrt(criterion(p, batch_y))
            step = p_step(loss, step)
            loss.backward()
            opt.step()
            opt.zero_grad()

            idx += bs

        rmse_ = validate_pd_prediction(hd_embs, ds_model, hd, bs, max_p, valid, epoch, False, task)

        if rmse_ < best_rmse:
            best_rmse = rmse_
            pickle.dump(ds_model, open(str(city) + '/Model/model_hp.pkl', 'wb'))
            print('[SAVED]')
        print(config.sep_width * "-")


def validate_pd_prediction(hd_embs, ds_model, hd, bs, max_p, valid, epoch, testing, task):
    for _ in range(2):
        print(config.sep_width * "-")

    if not testing:
        print('Validation - Epoch: ' + str(epoch + 1))

    else:
        print('Testing best model...')

    print('(Population Density prediction)')

    print(config.sep_width * "-")

    ds_model.eval()

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    idx = 0

    pred = []

    while idx < len(valid):

        ds_model.train()

        if idx > len(valid) - bs:
            batch_l = valid[idx:]

        else:
            batch_l = valid[idx:idx + bs]

        batch_global = []
        batch_local = []
        batch_p = []

        for element in batch_l:
            batch_global.append(torch.tensor(hd_embs[element]['global']))
            batch_local.append(torch.tensor(hd_embs[element]['local']))
            batch_p.append(torch.tensor([pos / 10 for pos in hd_embs[element]['pos']]))

        batch_global = torch.stack(batch_global)
        batch_local = torch.stack(batch_local)
        batch_p = torch.stack(batch_p)

        with torch.no_grad():
            p = ds_model(batch_global, batch_local, batch_p) * max_p

        if not p.shape:
            p = p.unsqueeze(0)

        pred.append(p)

        idx += bs

    pred = torch.cat(pred, 0)

    if task == 'hp':
        y = torch.tensor([hd[str(int(h) // 2)]['price'] for h in valid]).float()
    else:
        y = torch.tensor([hd_embs[h]['pop_density'] for h in valid]).float()

    mse_ = mse(pred, y).item()
    rmse_ = sqrt(mse_)
    l1_ = l1(pred, y).item()
    r2 = r2_score(pred, y).item()

    print('RMSE: ' + str(round(rmse_, 2)) + ' | MAE: ' + str(round(l1_, 2)) + ' | R2: ' + str(round(r2, 4)))

    print(config.sep_width * "-")

    if not testing:
        return rmse_
    if testing:
        return rmse_, l1_, r2


def split_data(df):
    train = df.iloc[:int(df.shape[0] / 10 * 6), :]
    valid = df.iloc[int(df.shape[0] / 10 * 6):int(df.shape[0] / 10 * 8), :]
    test = df.iloc[int(df.shape[0] / 10 * 8):, :]

    return train, valid, test


def random_performance(test, n_c, lu_data_u, classes_one_hot):
    pred = []
    y = []

    for i in range(len(test)):
        pred.append(random.choice(list(range(0, n_c))))
        y.append(classes_one_hot[lu_data_u[test[i]]])

    micro = f1_score(pred, y, average='micro')
    macro = f1_score(pred, y, average='macro')
    weighted = f1_score(pred, y, average='weighted')

    print('F1 (micro): ' + str(round(micro * 100, 2)) + '% | F1 (macro): ' + str(
        round(macro * 100, 2)) + '% | F1 (weighted): ' + str(round(weighted * 100, 2)) + '%')

    exit()


def make_categories(city, keys, hd):
    ud = UrbanData(city, buildings_u=False)
    cats = {}
    c = 0

    hd_embs = {}

    for k in keys:
        for e in hd[k]['nodes_id']:
            tags = ud.entities[e]['tags']
            if 'amenity' in tags:
                if tags['amenity'] not in cats:
                    cats[tags['amenity']] = c
                    c += 1

            elif 'shop' in tags:
                if 'shop' not in cats:
                    cats['shop'] = c
                    c += 1

            elif 'tourism' in tags:
                if 'tourism' not in cats:
                    cats['tourism'] = c
                    c += 1

            elif 'office' in tags:
                if tags['office'] not in cats:
                    cats[tags['office']] = c
                    c += 1

            elif 'leisure' in tags:
                if tags['leisure'] not in cats:
                    cats[tags['leisure']] = c
                    c += 1

    for k in keys:
        cat_list = [np.zeros(len(cats))]
        # for e in hd[str(int(k)//2)]['nodes']:
        for e in hd[k]['nodes_id']:
            tags = ud.entities[e]['tags']
            this_cat = np.zeros(len(cats))

            if 'amenity' in tags:
                this_cat[cats[tags['amenity']]] = 1

            elif 'shop' in tags:
                this_cat[cats['shop']] = 1

            elif 'tourism' in tags:
                this_cat[cats['tourism']] = 1

            elif 'office' in tags:
                this_cat[cats[tags['office']]] = 1

            elif 'leisure' in tags:
                this_cat[cats[tags['leisure']]] = 1

            cat_list.append(this_cat)

        cat_list = [sum(sub_list) / len(sub_list) for sub_list in zip(*cat_list)]
        hd_embs[k] = cat_list

    return hd_embs


def precompute_embs_lu(ud, city, keys, model):
    tok = Tokenizer(config.lm)
    max_distance = 500.0

    device = config.device
    model = model.to(device)
    spatial_aware_svi_generator = Spatial_Aware_Semantic_Consistency(temperature=0.5).to(device)
    gatedfusion_triple = GatedFusion(model.emb_size, 3).to(device)

    all_embs = {}
    model.eval()

    def process_element(element, i, total_keys):

        with torch.cuda.amp.autocast():
            ctx = ud.buildings_ctx[element]
            pois = torch.tensor(serialize_data([ud.entities[k] for k in ctx], tok, add_zero=True))
            pois_m = torch.tensor(np.where(pois != 0, 1, 0))
            pois = pois.to(device)
            pois_m = pois_m.to(device)

            emb = model.projection_p(
                torch.mean(torch.mean(model.lm(pois, attention_mask=pois_m)[0][:, :, :], 0), 0))

            image = torch.tensor(poly_to_raster(Polygon(ud.buildings_svi[element]['points']))).unsqueeze(0).float().to(
                device)
            image_emb = model.projection_img(model.cv(image).squeeze())

            s = torch.tensor(norm_surface(ud.buildings_svi[element]['area'])).unsqueeze(-1).float().to(device)
            s_emb = model.projection_s(s)

            poly_emb = (s_emb + image_emb) / 2

            poi_node = [element]
            poi = torch.tensor(serialize_data([ud.buildings_svi[k] for k in poi_node], tok, add_zero=True))
            poi_m = torch.tensor(np.where(poi != 0, 1, 0))
            poi = poi.to(device)
            poi_m = poi_m.to(device)

            poi_emb = model.projection_p(
                torch.mean(torch.mean(model.lm(poi, attention_mask=poi_m)[0][:, :, :], 0), 0))

            svi_memory = []
            dist_memory = []

            dist_list = ud.buildings_svi[element]['svi_dist']
            id_list = ud.buildings_svi[element]['svi']
            angle_list = ud.buildings_svi[element]['angle']

            if len(id_list) != 0:
                for k in range(len(id_list)):
                    east_emb, south_emb, west_emb, north_emb = compute_svi_emb(model, city, id_list[k])
                    svi_emb = compute_svi(angle_list[k], east_emb, south_emb, west_emb, north_emb)

                    svi_memory.append(svi_emb)
                    dist_memory.append(max_distance / (max_distance + dist_list[k]))

            svi_tensor = torch.stack(svi_memory).to(device)
            distance_tensor = torch.tensor(dist_memory).float().to(device)
            svi_emb = spatial_aware_svi_generator(poi_emb, svi_tensor, distance_tensor)

            poi_emb = gatedfusion_triple(poi_emb, svi_emb, poly_emb)

            result = {
                'ctx': emb.tolist(),
                'poi': poi_emb.tolist()
            }

        del pois, pois_m, emb, image, image_emb, s, s_emb, poi, poi_m, poi_emb
        del svi_tensor, distance_tensor, svi_emb
        torch.cuda.empty_cache()

        return element, result

    batch_size = 4
    for batch_start in range(0, len(keys), batch_size):
        batch_keys = keys[batch_start:batch_start + batch_size]

        with torch.no_grad():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_element = {
                    executor.submit(process_element, element, i, len(batch_keys)): element
                    for i, element in enumerate(batch_keys)
                }

                for future in concurrent.futures.as_completed(future_to_element):
                    element, result = future.result()
                    all_embs[element] = result

                    s = 'Precomputing building embeddings... (' + str(round((len(all_embs) / len(keys)) * 100, 2)) + '%)'
                    in_place_print(s)

        torch.cuda.empty_cache()

    print(config.flush)
    save(all_embs, ud.city + '/Data/building_embeddings.json')


def land_use_prediction(city, ud, model, training=True, cm=False):
    if not os.path.isfile(city + '/Data/land_use.json'):
        print('Preprocessing ' + city + ' land use data...')
        land_use_to_json(city)

    with open(city + '/Data/land_use.json') as f:
        lu_data = json.load(f)

    lu_data_u = {}

    for key in lu_data.keys():
        if key in ud.buildings_u.keys():
            lu_data_u[key] = lu_data[key]

    k_list = list(lu_data_u.keys())

    if not os.path.isfile(city + '/Data/building_embeddings.json'):
        precompute_embs_lu(ud, city, list(lu_data_u.keys()), model)

    classes_d = {}

    for k in k_list:
        if lu_data_u[k] not in classes_d:
            classes_d[lu_data_u[k]] = [k]
        else:
            classes_d[lu_data_u[k]].append(k)

    classes_one_hot = {}
    o = 0
    for k in classes_d:
        classes_one_hot[k] = o
        o += 1

    train, valid, test = [], [], []

    for c in classes_d:

        random.shuffle(classes_d[c])
        for k in classes_d[c][:int(len(classes_d[c]) / 10 * 6)]:
            train.append(k)

        for k in classes_d[c][int(len(classes_d[c]) / 10 * 6):int(len(classes_d[c]) / 10 * 8)]:
            valid.append(k)

        for k in classes_d[c][int(len(classes_d[c]) / 10 * 8):]:
            test.append(k)

    print('Loading precomputed embeddings...')
    with open(str(city) + '/Data/building_embeddings.json', encoding='utf-8') as f:
        b_embs = json.load(f)

    ds_model = LULinear(len(list(set(lu_data_u.values()))), model.emb_size)
    train_land_use_prediction(ud, b_embs, ds_model, lu_data_u, classes_one_hot, train, valid, test, training=training,
                              cm=cm)


def freq_dic(data, classes_one_hot):
    freq = {}
    tot = len(data)

    for sid in data:

        c = classes_one_hot[data[sid]]

        if c not in freq:
            freq[c] = 1
        else:
            freq[c] += 1

    for k, v in freq.items():
        freq[k] = 1 / (v / tot)

    return torch.tensor(np.array(list(freq.values()))).float()


def train_land_use_prediction(ud, b_embs, ds_model, lu_data_u, classes_one_hot, train, valid, test, training, cm=False):
    if not cm:
        print(config.sep_width * "-")
        print('Supervised Training')
        print('(Land use prediction)')

    opt = optim.Adam(params=ds_model.parameters(), lr=config.ds_lr)
    bs = config.ds_bs
    num_steps = (len(train) // bs) * config.ds_epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)
    criterion = nn.NLLLoss()

    best_f1 = 0

    for epoch in range(config.ds_epochs):

        if not training:
            break

        print_epoch(epoch + 1, config.ds_epochs)

        random.shuffle(train)
        opt.zero_grad()

        idx = 0
        step = 0

        while idx < len(train):

            ds_model.train()

            if idx > len(train) - bs:
                batch_l = train[idx:]
                batch_y = torch.tensor([classes_one_hot[lu_data_u[k]] for k in batch_l])

            else:
                batch_l = train[idx:idx + bs]
                batch_y = torch.tensor([classes_one_hot[lu_data_u[k]] for k in batch_l])

            batch_vis = []
            batch_ctx = []
            batch_pe = []

            for element in batch_l:

                pe = []
                if element in ud.buildings_u:
                    b_u = ud.buildings_u[element]
                    if 'lat' in b_u and 'lon' in b_u:
                        pe = PE((b_u['lat'], b_u['lon']), config.pe_size)

                if not len(pe):
                    pe = np.zeros(config.pe_size)

                if element in b_embs:
                    emb = torch.tensor(b_embs[element]['ctx'])
                    batch_ctx.append(emb)
                    poi_emb = torch.tensor(b_embs[element]['poi'])
                    batch_vis.append(poi_emb)
                    batch_pe.append(torch.tensor(pe).float())

            batch_vis = torch.stack(batch_vis)
            batch_ctx = torch.stack(batch_ctx)
            batch_pe = torch.stack(batch_pe)

            p = ds_model(batch_vis, batch_ctx, batch_pe)

            loss = criterion(p, batch_y)
            step = p_step(loss, step)
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()

            idx += bs

        f1_ = validate_land_use_prediction(ud, b_embs, ds_model, None, valid,
                                           [classes_one_hot[lu_data_u[k]] for k in valid], epoch, step, bs,
                                           testing=False)

        if f1_ > best_f1:
            best_f1 = f1_
            pickle.dump(ds_model, open(str(ud.city) + '/Model/model_land_use.pkl', 'wb'))

            with open(config.log_file_name, 'a') as out_f:
                print('[SAVED]', file=out_f)

            print('[SAVED]')

        print(config.sep_width * "-")

    with open(str(ud.city) + '/Model/model_land_use.pkl', 'rb') as f:
        ds_model = pickle.load(f)

    _ = validate_land_use_prediction(ud, b_embs, ds_model, list(classes_one_hot.keys()), test,
                                     [classes_one_hot[lu_data_u[k]] for k in test], None, None, bs, testing=True, cm=cm)


def validate_land_use_prediction(ud, b_embs, ds_model, c_names, valid, y, epoch, step, bs, testing=False, cm=False):
    for _ in range(2):
        print(config.sep_width * "-")

    if not testing:
        print('Validation - Epoch: ' + str(epoch + 1) + ' Step: ' + str(step))

    else:
        print('Testing best model...')

    print('(Land use prediction)')

    print(config.sep_width * "-")

    ds_model.eval()

    idx = 0

    pred = []

    while idx < len(valid):

        ds_model.train()

        if idx > len(valid) - bs:
            batch_l = valid[idx:]

        else:
            batch_l = valid[idx:idx + bs]

        batch_vis = []
        batch_ctx = []
        batch_pe = []

        with torch.no_grad():

            for element in batch_l:

                pe = []
                if element in ud.buildings_u:
                    b_u = ud.buildings_u[element]
                    if 'lat' in b_u and 'lon' in b_u:
                        pe = PE((b_u['lat'], b_u['lon']), config.pe_size)

                if not len(pe):
                    pe = np.zeros(config.pe_size)

                if element in b_embs:
                    emb = torch.tensor(b_embs[element]['ctx'])
                    batch_ctx.append(emb)
                    poi_emb = torch.tensor(b_embs[element]['poi'])
                    batch_vis.append(poi_emb)
                    batch_pe.append(torch.tensor(pe).float())

            batch_vis = torch.stack(batch_vis)
            batch_ctx = torch.stack(batch_ctx)
            batch_pe = torch.stack(batch_pe)

            batch_p = ds_model(batch_vis, batch_ctx, batch_pe)

        for p in batch_p:
            pred.append(torch.argmax(p).item())

        idx += bs

    micro = f1_score(pred, y, average='micro')
    macro = f1_score(pred, y, average='macro')
    accuracy = accuracy_score(y, pred)
    weighted = f1_score(pred, y, average='weighted')

    # if testing:
    #    print(f1_score(pred, y, average=None))

    print('F1 (macro): ' + str(round(macro * 100, 2)) + '% | F1 (weighted): ' + str(
        round(weighted * 100, 2)) + '% | Accuracy: ' + str(round(accuracy * 100, 2)) + '%')

    with open(config.log_file_name, 'a') as out_f:
        print('F1 (macro): ' + str(round(macro * 100, 2)) + '% | F1 (weighted): ' + str(
            round(weighted * 100, 2)) + '% | Accuracy: ' + str(round(accuracy * 100, 2)) + '%', file=out_f)

    if cm:
        show_cm(y, pred, c_names)

    return macro


def show_cm(y, pred, c_names):
    cm = confusion_matrix(y, pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=c_names, yticklabels=c_names,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.xticks(rotation=45)
    plt.xticks(fontsize=6)

    fig.tight_layout()
    plt.show()


downstream_functions = {'build_func': land_use_prediction, 'pop_density': population_density}
