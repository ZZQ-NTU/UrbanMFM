import numpy as np
import tifffile
import config
from itertools import combinations
from shapely.geometry import Point, LineString
from utils.pre_processing import *
from utils.errors import *
from utils.models import *


class UrbanData:

    def __init__(self, city, buildings_u=False):
        self.city = city

        with open(str(city) + '/Data/data.json', encoding='utf-8') as f:
            self.data = json.load(f)

        with open(str(city) + '/Data/entities.json', encoding='utf-8') as f:
            self.entities = json.load(f)

        with open(str(city) + '/Data/relations.json', encoding='utf-8') as f:
            self.relations = json.load(f)

        with open(str(city) + '/Data/polylines.json', encoding='utf-8') as f:
            self.polylines = json.load(f)

        with open(str(city) + '/Data/entities_context.json', encoding='utf-8') as f:
            self.poi = json.load(f)

        with open(str(city) + '/Data/region_context.json', encoding='utf-8') as f:
            self.region = json.load(f)

        with open(str(city) + '/Data/satellite_pair.json', encoding='utf-8') as f:
            self.sate_pair = json.load(f)

        with open(str(city) + '/Data/pop_density.json', encoding='utf-8') as f:
            self.pop = json.load(f)

        scene_file = pd.read_csv(str(city) + '/Data/scenes.csv', encoding='utf-8')
        self.id = scene_file[scene_file.columns[0]]
        self.scene = scene_file[scene_file.columns[3]]
        self.index = list(range(len(self.scene)))

        self.tif_path = str(city) + '/Data/satellite.tif'

        sampling_point = pd.read_csv(str(city) + '/Data/sampling_points.csv', encoding='utf-8')
        self.FID = sampling_point[sampling_point.columns[0]]
        self.lons = sampling_point[sampling_point.columns[1]]
        self.lats = sampling_point[sampling_point.columns[2]]

        if buildings_u:
            with open(str(city) + '/Data/buildings_untagged.json', encoding='utf-8') as f:
                self.buildings_u = json.load(f)

            with open(str(city) + '/Data/buildings_svi.json', encoding='utf-8') as f:
                self.buildings_svi = json.load(f)

            with open(str(city) + '/Data/buildings_context.json', encoding='utf-8') as f:
                self.buildings_ctx = json.load(f)


class Tokenizer:

    def __init__(self, lm):

        if lm == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config.lm_names[lm])
        elif lm == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(config.lm_names[lm])
        elif lm == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(config.lm_names[lm])
        elif lm == 'multi_bert':
            self.tokenizer = BertTokenizer.from_pretrained(config.lm_names[config.default_lm])
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.lm_names[config.default_lm])

    def tokenize_batch(self, data, add_zero=False):

        if not data:
            data.append(config.no_context)

        elif add_zero:
            data.append(config.no_context)

        tok_data = []
        for d in data:
            tok_d = self.tokenizer.tokenize('[CLS] ' + d + ' [SEP]')
            tok_data.append(tok_d)

        tag_len = max([len(d) for d in tok_data])

        ids_data = []
        for tok_d in tok_data:

            if len(tok_d) < tag_len:
                tok_d += ['[PAD]'] * (tag_len - len(tok_d))
            else:
                tok_d = tok_d[:tag_len]

            ids_data.append(self.tokenizer.convert_tokens_to_ids(tok_d))

        return ids_data


def load_poly_dictionary(city):
    with open(str(city) + '/Data/polyline_to_id.json', encoding='utf-8') as f:
        pd = json.load(f)

    return pd


def serialize_data(data, tok, add_zero=False):
    entities_tags = []

    for d in data:
        tags = d['tags']
        rq_tags = {k: tags[k] for k in osm_strings.use_tags if k in tags}
        s_tags = json_string(rq_tags)
        entities_tags.append(s_tags)

    entities_tags = np.array(tok.tokenize_batch(entities_tags, add_zero=add_zero))
    return entities_tags


def serialize_cat(data, voc):
    entities_tags = []

    for d in data:
        tags = d['tags']
        rq_tags = {k: tags[k] for k in osm_strings.use_tags if k in tags}
        s_tags = json_string(rq_tags)
        entities_tags.append(voc[s_tags])

    if not data:
        entities_tags.append(voc[config.no_context])

    return entities_tags


def prepare_sequences(data, entities, relations):
    queries = np.array(list(relations.keys()))[:20]
    w = (check_window(config.rel_window) - 1) // 2
    tok = Tokenizer(config.lm)

    anchor_l = []
    pos_l = []

    for c, query in enumerate(queries):

        seq = relations[query]
        seq = [str(x) for x in seq]

        for i in range(len(seq) - 2 * w):

            for j in [x for x in range(-w, w + 1) if x != 0]:

                if seq[i + w] in data and seq[i + w + j] in data:

                    anchor_ctx = serialize_data([entities[k] for k in data[seq[i + w]]['context']], tok, add_zero=False)
                    pos_ctx = serialize_data([entities[k] for k in data[seq[i + w + j]]['context']], tok,
                                             add_zero=False)

                    anchor_l.append(anchor_ctx)
                    pos_l.append(pos_ctx)

        s = 'Serializing sequences... (' + str(round((c + 1) / len(queries) * 100, 2)) + '%)'
        in_place_print(s)

    print(config.flush)

    return anchor_l, pos_l


def build_category_vocab(entities):
    c = 1
    vocab = {config.no_context: 0}

    for e in entities:

        tags = entities[e]['tags']
        rq_tags = {k: tags[k] for k in osm_strings.use_tags if k in tags}
        s_tags = json_string(rq_tags)

        if s_tags not in vocab:
            vocab[s_tags] = c
            c += 1

    return vocab


def write_step(city, step_loss):
    with open(str(city) + '/Model/loss.txt', 'w') as f:
        for s_l in step_loss:
            f.write(str(s_l) + '\n')


def save_model(city, model, e, task):
    pickle.dump(model, open(str(city) + '/Model/model.pkl', 'wb'))


def check_window(w):
    if w < 3:
        window_size_error()

    if not w % 2:
        return w - 1

    return w


def print_step(loss, step_loss, step):
    step += 1

    if config.device == 'cuda':
        l_item = loss.cpu().detach().numpy()
        if isinstance(l_item, np.ndarray):
            l_item = l_item.item()
    else:
        l_item = loss.item()

    step_loss.append(l_item)

    print('Step ' + str(step) + ' Loss: ' + str(round(l_item, 4)))

    return step


def p_step(loss, step):
    step += 1

    if config.device == 'cuda':
        l_item = loss.cpu().detach().numpy()
        if isinstance(l_item, np.ndarray):
            l_item = l_item.item()  # Convert NumPy array to a scalar
    else:
        l_item = loss.item()

    with open(config.log_file_name, 'a') as out_f:
        print('Step ' + str(step) + ' Loss: ' + str(round(l_item, 4)), file=out_f)

    print('Step ' + str(step) + ' Loss: ' + str(round(l_item, 4)))

    return step


def make_relation_voc(relations):
    way_voc = {}

    for key in relations:

        r = relations[key]

        for way in r:

            w = str(way)

            if w not in way_voc:
                way_voc[w] = 1
            else:
                way_voc[w] += 1

    max_value = max([way_voc[k] for k in way_voc])

    for way in way_voc:
        way_voc[way] /= max_value

    return way_voc


def u_sampling(ud, poly_queries, idx):
    random.shuffle(poly_queries)
    used_c = []
    queries = []

    for p_q in poly_queries:

        tags = string_of_tags(ud.entities[p_q])
        if tags not in used_c:
            used_c.append(tags)
            queries.append(p_q)

            if len(queries) == idx:
                return queries

    i = len(poly_queries) - 1
    while len(queries) < idx:
        queries.append(poly_queries[i])
        i -= 1

    return queries


def train_SS_im(ud, model):
    print(config.sep_width * "-")
    print('Self-supervised Training')
    print('(Infomax)')

    queries = np.array(list(ud.data.keys()))
    tok = Tokenizer(config.lm)
    opt = optim.Adam(params=model.parameters(), lr=config.lr)
    bs = config.bs
    num_steps = (len(queries) // bs) // 5  # * epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

    step_loss = []

    device = config.device
    info_nce = InfoNCE(config.index)

    for epoch in range(config.n_epochs):

        print_epoch(epoch + 1, config.n_epochs)

        model.to(device)

        step = 0

        # noinspection PyTypeChecker
        random.shuffle(queries)
        idx = 0
        loss = torch.tensor(0.)
        opt.zero_grad()

        while idx < len(queries):

            if idx > len(queries) - config.index:
                # query = queries[idx:]
                break

            else:
                query = queries[idx:idx + config.index]

            ctx = []
            query_p = []

            for q in query:
                context = ud.data[q]['context']
                if not context:
                    query_p.append(serialize_data([], tok))

                else:
                    r = random.randint(0, len(context))
                    if r == len(context):
                        query_p.append(serialize_data([], tok))

                    else:
                        if context[r] in ud.entities:
                            query_p.append(serialize_data([ud.entities[context[r]]], tok))
                        else:
                            query_p.append(serialize_data([], tok))

                ctx.append(serialize_data([ud.entities[k] for k in context if k in ud.entities], tok, add_zero=True))

            p_line_embs, ctx_embs = model(ud, query_p, ctx, task='info_max')  # 64 x 768

            p_line_embs = p_line_embs.cpu()
            ctx_embs = ctx_embs.cpu()
            loss += info_nce(p_line_embs, ctx_embs)

            idx += config.index

            if idx % bs == 0:
                step = print_step(loss, step_loss, step)
                loss.backward()
                opt.step()
                scheduler.step()
                loss = torch.tensor(0.)
                opt.zero_grad()

            if idx > config.m_index:
                break

        write_step(ud.city, step_loss)
        save_model(ud.city, model.cpu(), epoch + 1, 'im')

    print('\n\n')


def train_SS_svi(ud, model):
    print(config.sep_width * "-")
    print('Self-supervised Training')
    print('(Street View Image Contrastive Learning)')

    scene_queries = list(ud.scene)
    id_queries = list(ud.id)
    index_queries = list(ud.index)
    tok = Tokenizer(config.lm)
    opt = optim.Adam(params=model.parameters(), lr=config.lr / 10)
    bs = config.bs_svi
    num_steps = (len(index_queries) // bs) // 5  # * epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

    step_loss = []

    device = config.device
    info_nce = InfoNCE(config.index_svi)

    for epoch in range(config.n_epochs):
        print_epoch(epoch + 1, config.n_epochs)

        model.to(device)

        step = 0

        random.shuffle(index_queries)
        idx = 0
        loss = torch.tensor(0.)
        opt.zero_grad()

        # print(len(id_queries))

        while idx < len(index_queries):

            if idx > len(index_queries) - config.index_svi:
                break

            else:
                query = index_queries[idx:idx + config.index_svi]

            text = []
            visual = []

            for q in query:
                scene = 'scene: ' + scene_queries[int(q)]
                text.append(np.array(tok.tokenize_batch([scene], add_zero=True)))
                visual.append(id_queries[int(q)])

            svi_embs, pos_embs = model(ud, text, visual, task='svi')
            svi_emb = svi_embs.cpu()
            pos_emb = pos_embs.cpu()

            loss += info_nce(svi_emb, pos_emb)

            idx += config.index_svi

            if idx % bs == 0:
                step = print_step(loss, step_loss, step)
                loss.backward()
                opt.step()
                scheduler.step()
                loss = torch.tensor(0.)
                opt.zero_grad()

            if idx > config.m_index:
                break

        write_step(ud.city, step_loss)
        save_model(ud.city, model.cpu(), epoch + 1, 'svi')

    print('\n\n')


def train_SS_poi(ud, model):
    print(config.sep_width * "-")
    print('Self-supervised Training')
    print('(POI-level Triplet Contrastive Learning)')

    queries = np.array(list(ud.poi.keys()))

    entity_queries = []
    for k in range(len(queries)):
        neighbour = ud.poi[queries[k]]['neighbour']
        if len(neighbour) > 2:
            entity_queries.append(queries[k])

    tok = Tokenizer(config.lm)
    opt = optim.Adam(params=model.parameters(), lr=config.lr / 10)
    bs = config.bs_poi
    num_steps = (len(entity_queries) // bs) // 5  # * epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

    step_loss = []
    device = config.device

    info_nce = InfoNCE(config.index_poi)

    for epoch in range(config.n_epochs):
        print_epoch(epoch + 1, config.n_epochs)

        model.to(device)

        # epsilon = 1.0
        step = 0

        random.shuffle(entity_queries)
        idx = 0
        loss = torch.tensor(0.)
        opt.zero_grad()

        while idx < len(entity_queries):
            
            if idx > len(entity_queries) - config.index_poi:
                break

            else:
                query = entity_queries[idx:idx + config.index_poi]

            G_list = []

            for q in query:

                G = nx.Graph()

                neighbour = ud.poi[q]['neighbour']
                neigh_dist = ud.poi[q]['neigh_dist']

                G.add_node(q, type='POI', value=serialize_data([ud.entities[q]], tok))

                for i in range(len(neighbour)):
                    G.add_node(neighbour[i], type='neighbour', value=serialize_data([ud.entities[neighbour[i]]], tok))
                    G.add_edge(q, neighbour[i], type='sur_neigh', weight=1 / neigh_dist[i])

                G_list.append(G)

            embs, pos_embs = model(ud, G_list, None, task='poi_triplet')

            embs = embs.cpu()
            pos_embs = pos_embs.cpu()

            loss += info_nce(embs, pos_embs)

            idx += config.index_poi

            if idx % bs == 0:
                step = print_step(loss, step_loss, step)
                loss.backward()
                opt.step()
                scheduler.step()
                loss = torch.tensor(0.)
                opt.zero_grad()

            if idx > config.m_index:
                break

        write_step(ud.city, step_loss)
        save_model(ud.city, model.cpu(), epoch + 1, 'poi_triplet')

    print('\n\n')


def train_SS_sate(ud, model):
    print(config.sep_width * "-")
    print('Self-supervised Training')
    print('(Satellite Image Contrastive Learning)')

    queries = np.array(list(ud.sate_pair.keys()))
    opt = optim.Adam(params=model.parameters(), lr=config.lr / 10)
    bs = config.bs_sate
    num_steps = (len(queries) // bs) // 5  # * epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

    step_loss = []
    device = config.device

    info_nce = InfoNCE(config.index_sate)

    with tifffile.TiffFile(ud.tif_path) as tif:
        sate_img = tif.pages[0].geotiff_tags
        tif_image = tif.asarray()

        for epoch in range(config.n_epochs):
            print_epoch(epoch + 1, config.n_epochs)

            model.to(device)

            step = 0

            random.shuffle(queries)
            idx = 0
            loss = torch.tensor(0.)
            opt.zero_grad()

            while idx < len(queries):

                if idx > len(queries) - config.index_sate:
                    break

                else:
                    query = queries[idx:idx + config.index_sate]

                sate = []
                pos = []

                for q in query:
                    sate_i = ud.pop[str(ud.sate_pair[q]['pair'][0])]
                    sate_j = ud.pop[str(ud.sate_pair[q]['pair'][1])]

                    region_i = Polygon([[p[1], p[0]] for p in sate_i['polygon']])
                    region_j = Polygon([[p[1], p[0]] for p in sate_j['polygon']])

                    coordinates_i = []
                    coordinates_j = []

                    for coord in region_i.exterior.coords:
                        coordinates_i.append((coord[0], coord[1]))
                    for coord in region_j.exterior.coords:
                        coordinates_j.append((coord[0], coord[1]))

                    sate_emb = extract_image_from_coords(sate_img, tif_image, coordinates_i)
                    pos_emb = extract_image_from_coords(sate_img, tif_image, coordinates_j)

                    sate.append(sate_emb)
                    pos.append(pos_emb)

                sate_embs, pos_embs = model(ud, sate, pos, task='sate')

                sate_embs = sate_embs.cpu()
                pos_embs = pos_embs.cpu()

                loss += info_nce(sate_embs, pos_embs)

                idx += config.index_sate

                if idx % bs == 0:
                    step = print_step(loss, step_loss, step)
                    loss.backward()
                    opt.step()
                    scheduler.step()
                    loss = torch.tensor(0.)
                    opt.zero_grad()

                if idx > config.m_index:
                    break

            write_step(ud.city, step_loss)
            save_model(ud.city, model.cpu(), epoch + 1, 'sate')

    print('\n\n')


def train_SS_region(ud, model):
    print(config.sep_width * "-")
    print('Self-supervised Training')
    print('(Region-level Triplet Contrastive Learning)')

    queries = np.array(list(ud.region.keys()))

    opt = optim.Adam(params=model.parameters(), lr=config.lr / 10)
    tok = Tokenizer(config.lm)
    bs = config.bs_region
    num_steps = (len(queries) // bs) // 5  # * epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

    step_loss = []
    device = config.device

    info_nce = InfoNCE(config.index_region)

    with tifffile.TiffFile(ud.tif_path) as tif:
        sate_img = tif.pages[0].geotiff_tags
        tif_image = tif.asarray()

        for epoch in range(config.n_epochs):
            print_epoch(epoch + 1, config.n_epochs)

            model.to(device)

            step = 0

            random.shuffle(queries)
            idx = 0
            loss = torch.tensor(0.)
            opt.zero_grad()

            while idx < len(queries):

                if idx > len(queries) - config.index_region:
                    break

                else:
                    query = queries[idx:idx + config.index_region]

                G_list = []

                for q in query:

                    G = nx.Graph()

                    neighbour = ud.region[q]['neighbour']
                    pois = ud.region[q]['entity']
                    sate = ud.region[q]
                    region = Polygon([[p[1], p[0]] for p in sate['polygon']])

                    coordinates = []
                    poi_list = []
                    poi_id = []

                    for coord in region.exterior.coords:
                        coordinates.append((coord[0], coord[1]))

                    for m in range(len(pois)):
                        poi = torch.tensor(
                            serialize_data([ud.entities[pois[m]]], tok))
                        poi_list.append(poi)
                        poi_id.append(pois[m])

                    G.add_node(q, type='region', value=extract_image_from_coords(sate_img, tif_image, coordinates),
                               poi_value=poi_list, poi_id=poi_id)

                    for i in range(len(neighbour)):
                        sate_i = ud.pop[neighbour[i]]
                        pois_i = ud.region[neighbour[i]]['entity']
                        region_i = Polygon([[p[1], p[0]] for p in sate_i['polygon']])

                        coordinates_i = []
                        poi_list_i = []
                        poi_id_i = []

                        for coord in region_i.exterior.coords:
                            coordinates_i.append((coord[0], coord[1]))

                        for n in range(len(pois_i)):
                            poi_i = torch.tensor(
                                serialize_data([ud.entities[pois_i[n]]], tok))
                            poi_list_i.append(poi_i)
                            poi_id_i.append(pois_i[n])
                        G.add_node(neighbour[i], type='neighbour',
                                   value=extract_image_from_coords(sate_img, tif_image, coordinates_i),
                                   poi_value=poi_list_i, poi_id=poi_id_i)
                        G.add_edge(q, neighbour[i], type='sur_neigh')

                    G_list.append(G)

                region_embs, pos_embs = model(ud, G_list, None, task='region_triplet')

                region_embs = region_embs.cpu()
                pos_embs = pos_embs.cpu()

                loss += info_nce(region_embs, pos_embs)

                idx += config.index_region

                if idx % bs == 0:
                    step = print_step(loss, step_loss, step)
                    loss.backward()
                    opt.step()
                    scheduler.step()
                    loss = torch.tensor(0.)
                    opt.zero_grad()

                if idx > config.r_index:
                    break

            write_step(ud.city, step_loss)
            save_model(ud.city, model.cpu(), epoch + 1, 'region_triplet')

    print('\n\n')
