import torch
from torch import nn, optim
import torch.nn.functional as F
import copy
# import timm
from transformers import DistilBertModel, RobertaModel
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
import torchvision.transforms as transforms
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx, to_networkx
from PIL import Image
# from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm

import config
from utils.pre_processing import *


def PE(pos, dim):
    if dim % 4:
        print("Error: 'dim' must be a multiple of 4")
        exit()

    p_enc = []

    for i in range(0, dim // 2, 2):

        for loc in pos:
            w_k = config.lambda_ / pow(10000, i / (dim // 2))
            p_enc.append(config.lambda_ * sin(loc * w_k))
            p_enc.append(config.lambda_ * cos(loc * w_k))

    return np.array(p_enc)


class HPRegressor(nn.Module):

    def __init__(self, emb_size):
        super().__init__()

        self.emb_size = emb_size
        # self.dropout = config.dropout

        self.regressor = nn.Sequential(
            nn.Linear(self.emb_size + config.pe_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        """
        self.local_trans = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.LayerNorm(self.emb_size),
            nn.Dropout(self.dropout)
        )

        self.global_trans = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.LayerNorm(self.emb_size),
            nn.Dropout(self.dropout)
        )
        """

        self.gate = nn.Sequential(
            # nn.Linear(self.hidden_size, 1),
            # nn.Softmax(dim=0)
            nn.Linear(self.emb_size * 2, 1),
            nn.Sigmoid()
        )

        self.filtration_gate = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.ReLU()
        )

        """

        self.gate = nn.Sequential(
            nn.Linear(self.emb_size, 1),
            nn.Softmax(dim=0)
        )

        self.filtration_gate = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.ReLU()
        )
        """

    def forward(self, global_emb, local_emb, pos_emb):
        # local_emb = self.local_trans(local_emb.unsqueeze(0))
        # global_emb = self.global_trans(global_emb.unsqueeze(0))
        # local_emb = local_emb.unsqueeze(0)
        # global_emb = global_emb.unsqueeze(0)

        attn_gate = self.gate(torch.cat([global_emb, local_emb], dim=-1))
        fusion = attn_gate * global_emb + (1 - attn_gate) * local_emb
        # attn_gate = self.gate(torch.cat([global_emb, local_emb], dim=0)).squeeze()
        # fusion = (global_emb - local_emb) * attn_gate[0].unsqueeze(-1) + local_emb
        urban_emb = self.filtration_gate(torch.cat([global_emb, fusion], dim=-1)).squeeze(0)

        # urban_emb = self.gated_fusion(local_emb.unsqueeze(0), global_emb.unsqueeze(0), None).squeeze()

        return self.regressor(torch.cat([urban_emb, pos_emb], dim=1)).squeeze()


class LULinear(nn.Module):

    def __init__(self, n_classes, poly_emb_size):
        super().__init__()

        self.poly_emb_size = poly_emb_size
        self.n_classes = n_classes

        self.polyline_sequential = nn.Sequential(
            nn.Linear(self.poly_emb_size * 2 + config.pe_size, 256),
            # nn.Linear(self.poly_emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes)
        )

    def forward(self, x, ctx, pe):

        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        if len(ctx.shape) < 2:
            ctx = ctx.unsqueeze(0)

        if len(pe.shape) < 2:
            pe = pe.unsqueeze(0)

        x = self.polyline_sequential(torch.cat([x, ctx, pe], dim=1))
        # x = self.polyline_sequential(x)

        return F.log_softmax(x, dim=1)


class GeoVectorsClassifier(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        self.sequential = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes)
        )

    def forward(self, x):
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        x = self.sequential(x)

        return F.log_softmax(x, dim=1)


class GeoVectorsRegressor(nn.Module):

    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        x = self.sequential(x)

        return x.squeeze()


class AVGSpeedLinear(nn.Module):

    def __init__(self, way_emb_size):
        super().__init__()

        self.way_emb_size = way_emb_size

        self.way_emb_linear = nn.Linear(self.way_emb_size, 128)
        # self.way_emb_linear = nn.Linear(128, 128)

        hw_emb = nn.Embedding(osm_strings.hw_emb, 128)
        lane_emb = nn.Embedding(osm_strings.lane_emb, 128)
        ms_emb = nn.Embedding(osm_strings.ms_emb, 128)

        self.rel_emb = nn.Linear(1, 128)
        self.emb_list = [hw_emb, lane_emb, ms_emb]

        self.polyline_sequential = nn.Sequential(
            nn.Linear(640 + config.pe_size // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, way_emb, f_emb, pe, rel):
        rel_emb = self.rel_emb(rel.unsqueeze(-1))

        way_emb = self.way_emb_linear(way_emb).squeeze()

        emb = [way_emb, pe, rel_emb]
        # emb = []
        for i in range(len(self.emb_list)):
            emb.append(self.emb_list[i](f_emb[:, i]))

        # emb = [way_emb, rel_emb]
        polyline_emb = torch.cat(emb, 1)

        return self.polyline_sequential(polyline_emb).squeeze()


class SSModel(nn.Module):

    def __init__(self, city, lm, emb_size):
        super().__init__()

        self.city = city

        self.lm_name = lm
        if lm not in config.lm_names:
            self.lm_name = config.default_lm

        self.lm = None
        self.get_lm()
        self.emb_size = emb_size

        resnet18 = torch.hub.load(config.torch_vision, config.vision_model)
        self.cv = torch.nn.Sequential(*(list(resnet18.children())[:-1]))

        # self.gatedfusion_triple = GatedFusion(self.emb_size, 3)
        # self.gatedfusion_double = GatedFusion(self.emb_size, 2)

        self.lm_hidden_size = config.lm_hidden_sizes[self.lm_name]
        self.vhs = config.vision_hidden_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.attention_linear = nn.Linear(self.lm_hidden_size, 1)

        self.attention_sequential = nn.Sequential(
            nn.Linear(self.lm_hidden_size, self.lm_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lm_hidden_size, 1)
        )

        self.projection_s = nn.Sequential(
            nn.Linear(1, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # self.context_proj = nn.Linear(self.lm_hidden_size, emb_size)
        self.projection_p = nn.Sequential(
            nn.Linear(self.lm_hidden_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        self.projection_img = nn.Linear(self.vhs, emb_size)

        self.fc = nn.Sequential(
            nn.Linear(self.lm_hidden_size, self.lm_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.lm_hidden_size // 2, 1)
        )

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        self.linear = nn.Linear(2 * emb_size, emb_size)

    def forward(self, cd, queries, keys, surfaces=None, device='cuda', task='info_max'):
        query_emb = []
        key_emb = []

        if task == 'raster':

            self.lm.eval()

            query, query_m = q_tensor_mask(queries)
            keys = torch.tensor(keys).float()
            surfaces = torch.tensor(surfaces).unsqueeze(-1)

            with torch.no_grad():
                query_emb = self.projection_p(torch.mean(self.lm(query, attention_mask=query_m)[0][:, :, :], 1))

            key_emb = self.projection_img(self.cv(keys).squeeze())
            s_emb = self.projection_s(surfaces)
            key_emb = (key_emb + s_emb) / 2

        elif task == 'svi':

            self.lm.eval()

            query_emb = []
            key_emb = []

            for i in range(len(queries)):
                query, query_m = q_tensor_mask(queries[i])
                query = query.to(device)
                query_m = query_m.to(device)

                with torch.no_grad():
                    q_enc = self.lm(query, attention_mask=query_m)[0][:, :, :]
                    q_emb = self.projection_p(torch.mean(torch.mean(q_enc, 0), 0))

                e_emb, s_emb, w_emb, n_emb = compute_svi_emb(self, self.city, keys[i])

                east_emb = self.linear(torch.cat([q_emb, e_emb], 0))
                south_emb = self.linear(torch.cat([q_emb, s_emb], 0))
                west_emb = self.linear(torch.cat([q_emb, w_emb], 0))
                north_emb = self.linear(torch.cat([q_emb, n_emb], 0))

                k = random.randint(0, 5)
                if k == 0:
                    query_emb.append(east_emb)
                    key_emb.append(south_emb)
                elif k == 1:
                    query_emb.append(east_emb)
                    key_emb.append(west_emb)
                elif k == 2:
                    query_emb.append(east_emb)
                    key_emb.append(north_emb)
                elif k == 3:
                    query_emb.append(south_emb)
                    key_emb.append(west_emb)
                elif k == 4:
                    query_emb.append(south_emb)
                    key_emb.append(north_emb)
                else:
                    query_emb.append(west_emb)
                    key_emb.append(north_emb)

            query_emb = torch.stack(query_emb)
            key_emb = torch.stack(key_emb)

        elif task == 'poi_triplet':

            self.lm.eval()

            G_list = queries
            query_emb = []
            key_emb = []

            GNN_model = GAT(in_channels=768, out_channels=768, task=task).to(device)
            GNN_model.eval()

            processed_Gs = []
            processed_pos_Gs = []

            def process_single_graph(G, device):
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    data, G_pos = process_local_graph(self, G, device)
                    G_pos =  remove_random_poi_nodes(G_pos)
                    data_pos = process_pos_local_graph(self, cd, G_pos, device)
                    return data, data_pos

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_graph = {executor.submit(process_single_graph, G, device): G for G in G_list}

                for future in concurrent.futures.as_completed(future_to_graph):
                    data, data_pos = future.result()
                    processed_Gs.append(data)
                    processed_pos_Gs.append(data_pos)

            torch.cuda.synchronize(device=device)

            for data, data_pos in zip(processed_Gs, processed_pos_Gs):
                emb = GNN_model(data.to(device))
                pos_emb = GNN_model(data_pos.to(device))

                q_emb = torch.mean(emb, dim=0)
                k_emb = torch.mean(pos_emb, dim=0)

                query_emb.append(q_emb)
                key_emb.append(k_emb)

            # 最终将结果堆叠
            query_emb = torch.stack(query_emb)
            key_emb = torch.stack(key_emb)

        elif task == 'sate':

            query_emb = []
            key_emb = []

            for i in range(len(queries)):

                satellite_image = queries[i]

                if satellite_image.size > 0:
                    satellite_image = Image.fromarray(satellite_image)
                    if satellite_image.mode != 'RGB':
                        satellite_image = satellite_image.convert('RGB')
                    satellite_input_tensor = self.preprocess(satellite_image).to(device)
                    satellite_input_batch = satellite_input_tensor.unsqueeze(0)
                    sate_emb = self.projection_img(
                        self.cv(satellite_input_batch).squeeze()).to(device)
                else:
                    sate_emb = torch.zeros(self.emb_size).to(device)

                satellite_image = keys[i]

                if satellite_image.size > 0:
                    satellite_image = Image.fromarray(satellite_image)
                    if satellite_image.mode != 'RGB':
                        satellite_image = satellite_image.convert('RGB')
                    satellite_input_tensor = self.preprocess(satellite_image).to(device)
                    satellite_input_batch = satellite_input_tensor.unsqueeze(0)
                    pos_emb = self.projection_img(
                        self.cv(satellite_input_batch).squeeze()).to(device)
                else:
                    pos_emb = torch.zeros(self.emb_size).to(device)

                query_emb.append(sate_emb)
                key_emb.append(pos_emb)

            query_emb = torch.stack(query_emb)
            key_emb = torch.stack(key_emb)

        elif task == "region_triplet":

            self.lm.eval()

            G_list = queries
            query_emb = []
            key_emb = []

            GNN_model = GAT(in_channels=768, out_channels=768, task=task).to(device)
            GNN_model.eval()

            processed_Gs = []
            processed_pos_Gs = []

            def process_single_graph(G, device):
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    data, G_pos = process_global_graph(self, G, device)
                    data_pos = process_pos_global_graph(self, cd, G_pos, device)
                    return data, data_pos

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_graph = {executor.submit(process_single_graph, G, device): G for G in G_list}

                for future in concurrent.futures.as_completed(future_to_graph):
                    data, data_pos = future.result()
                    processed_Gs.append(data)
                    processed_pos_Gs.append(data_pos)

            torch.cuda.synchronize(device=device)

            for data, data_pos in zip(processed_Gs, processed_pos_Gs):
                emb = GNN_model(data.to(device))
                pos_emb = GNN_model(data_pos.to(device))

                q_emb = torch.mean(emb, dim=0)
                k_emb = torch.mean(pos_emb, dim=0)

                query_emb.append(q_emb)
                key_emb.append(k_emb)

            # 最终将结果堆叠
            query_emb = torch.stack(query_emb)
            key_emb = torch.stack(key_emb)

        else:

            for i in range(len(queries)):

                query = queries[i]
                key = keys[i]

                query, query_m, key, key_m = qk_tensor_mask(query, key)
                query = query.to(device)
                query_m = query_m.to(device)
                key = key.to(device)
                key_m = key_m.to(device)

                if task == 'info_max':

                    q_enc = self.lm(query, attention_mask=query_m)[0][:, :, :]
                    q_emb = self.projection_p(torch.mean(torch.mean(q_enc, 0), 0))
                    query_emb.append(q_emb)

                    k_enc = self.lm(key, attention_mask=key_m)[0][:, :, :]
                    k_emb = self.projection_p(torch.mean(torch.mean(k_enc, 0), 0))
                    key_emb.append(k_emb)

                elif task == 'relation_seq':

                    self.lm.eval()

                    with torch.no_grad():
                        q_emb = self.projection_p(torch.mean(self.lm(query, attention_mask=query_m)[0][:, :, :], 1))
                        k_emb = self.projection_p(torch.mean(self.lm(key, attention_mask=key_m)[0][:, :, :], 1))

                    q_attn = F.softmax(self.attention_sequential(q_emb), 0)
                    q_emb = torch.sum(q_emb * q_attn, 0)
                    # q_emb = torch.mean(q_emb, 0)

                    k_attn = F.softmax(self.attention_sequential(k_emb), 0)
                    k_emb = torch.sum(k_emb * k_attn, 0)
                    # k_emb = torch.mean(k_emb, 0)

                    query_emb.append(q_emb)
                    key_emb.append(k_emb)

            query_emb = torch.stack(query_emb)
            key_emb = torch.stack(key_emb)

        return query_emb, key_emb

    def get_lm(self):

        if self.lm_name == 'bert':
            self.lm = BertModel.from_pretrained(config.lm_names[self.lm_name])
        elif self.lm_name == 'roberta':
            self.lm = RobertaModel.from_pretrained(config.lm_names[self.lm_name])
        elif self.lm_name == 'distilbert':
            self.lm = DistilBertModel.from_pretrained(config.lm_names[self.lm_name])
        else:
            self.lm = BertModel.from_pretrained(config.lm_names[self.lm_name])

    def predict_context(self, query_emb, key_emb):

        p0 = torch.abs(torch.sub(query_emb[0], key_emb[0]))
        p1 = torch.abs(torch.sub(query_emb[1], key_emb[1]))

        n0 = torch.abs(torch.sub(query_emb[0], key_emb[1]))
        n1 = torch.abs(torch.sub(query_emb[1], key_emb[0]))

        p0 = self.sigmoid(self.fc(p0))
        p1 = self.sigmoid(self.fc(p1))
        n0 = self.sigmoid(self.fc(n0))
        n1 = self.sigmoid(self.fc(n1))

        return p0, p1, n0, n1


def device_as(t1, t2):
    return t1.to(t2.device)


def q_tensor_mask(query):
    query = torch.tensor(query)

    query_m = np.where(query != 0, 1, 0)
    query_m = torch.tensor(query_m)

    return query, query_m


def qk_tensor_mask(query, key):
    query = torch.tensor(query)
    key = torch.tensor(key)

    query_m = np.where(query != 0, 1, 0)
    query_m = torch.tensor(query_m)

    key_m = np.where(key != 0, 1, 0)
    key_m = torch.tensor(key_m)

    return query, query_m, key, key_m


def compute_poly_emb(model, raster, surface):

    image = torch.tensor(raster).unsqueeze(0).float()
    image_emb = model.projection_img(model.cv(image).squeeze())
    s = torch.tensor(surface).unsqueeze(-1)
    s_emb = model.projection_s(s)
    poly_emb = image_emb + s_emb / 2

    return poly_emb


def process_local_graph(model, G, device='cude'):
    node_list = list(G.nodes)

    for i in range(len(node_list)):
        node_id = node_list[i]

        value, value_m = q_tensor_mask(G.nodes[node_id]['value'])
        value = value.to(device)
        value_m = value_m.to(device)

        # with torch.no_grad():
        v_enc = model.lm(value, attention_mask=value_m)[0][:, :, :]
        v_emb = model.projection_p(torch.mean(torch.mean(v_enc, 0), 0))

        G.nodes[node_id]['value'] = torch.tensor(v_emb.tolist())

    data = from_networkx(G)

    return data, G


def process_pos_local_graph(model, cd, G_pos, device='cuda'):
    svi_generator = Spatial_Aware_Semantic_Consistency(temperature=0.5)

    max_dist = 500.0

    pos_node_list = list(G_pos.nodes)
    for i in range(len(pos_node_list)):
        pos_node_id = str(pos_node_list[i])

        if G_pos.nodes[pos_node_id]['type'] != 'road':
            if cd.entities[pos_node_id]['type'] == 'polygon' and len(
                    cd.entities[pos_node_id]['points']) >= 3:
                random_k = random.randint(0, 2)
            else:
                random_k = random.randint(0, 1)

            if random_k == 0:
                G_pos.nodes[pos_node_id]['value'] = G_pos.nodes[pos_node_id]['value']
            elif random_k == 1:
                poi_emb = G_pos.nodes[pos_node_id]['value'].to(device)

                context = cd.poi[pos_node_id]['svi']
                distance = cd.poi[pos_node_id]['svi_dist']
                angle = cd.poi[pos_node_id]['angle']

                svi_memory = []
                distance_memory = []

                with torch.no_grad():
                    for m in range(len(context)):
                        east_svi_emb, south_svi_emb, west_svi_emb, north_svi_emb = compute_svi_emb(
                            model, model.city, context[m])
                        svi_emb = compute_svi(
                            angle[m], east_svi_emb, south_svi_emb, west_svi_emb, north_svi_emb)

                        if isinstance(svi_emb, tuple):
                            svi_emb = svi_emb[0]
                        svi_memory.append(svi_emb)
                        distance_memory.append(max_dist / (distance[m] + max_dist))

                    svi_tensor = torch.stack(svi_memory).to(device)
                    distance_tensor = torch.tensor(distance_memory).to(device)
                    svi_emb = svi_generator(poi_emb, svi_tensor, distance_tensor)

                G_pos.nodes[pos_node_id]['value'] = torch.tensor(svi_emb.tolist())

            else:
                with torch.no_grad():
                    entity = cd.entities[str(pos_node_id)]
                    raster = poly_to_raster(Polygon(entity['points']))
                    if not len(raster):
                        continue
                    image = torch.tensor(raster).unsqueeze(0).float().to(device)
                    image_emb = model.projection_img(model.cv(image).squeeze())
                    s = torch.tensor(norm_surface(entity['area'])).unsqueeze(-1).to(device)
                    s_emb = model.projection_s(s)
                    poly_emb = image_emb + s_emb / 2

                G_pos.nodes[pos_node_id]['value'] = torch.tensor(poly_emb.tolist())

        if isinstance(G_pos.nodes[pos_node_list[i]]['value'], np.ndarray):
            G_pos.nodes[pos_node_list[i]]['value'] = torch.tensor(G_pos.nodes[pos_node_list[i]]['value'])

    data_pos = from_networkx(G_pos)

    return data_pos


def process_global_graph(model, G, device='cuda'):
    node_list = list(G.nodes)

    for i in range(len(node_list)):
        node_id = str(node_list[i])
        # print(i)
        satellite_image = G.nodes[node_id]['value']

        if satellite_image.size > 0:
            satellite_image = Image.fromarray(satellite_image)
            if satellite_image.mode != 'RGB':
                satellite_image = satellite_image.convert('RGB')
            # with torch.no_grad():
            satellite_input_tensor = model.preprocess(satellite_image).to(device)
            satellite_input_batch = satellite_input_tensor.unsqueeze(0)
            G.nodes[node_id]['value'] = torch.tensor(model.projection_img(
                model.cv(satellite_input_batch).squeeze()).tolist()).to(device)
        else:
            G.nodes[node_id]['value'] = torch.zeros(model.emb_size).to(device)

        if isinstance(G.nodes[node_id]['value'], np.ndarray):
            G.nodes[node_id]['value'] = torch.tensor(G.nodes[node_id]['value']).to(device)

    data = from_networkx(G)

    return data, G


def process_pos_global_graph(model, cd, G_pos, device='cuda'):
    pos_node_list = list(G_pos.nodes)
    svi_generator = Spatial_Aware_Semantic_Consistency(temperature=0.5)
    polygon_encoder = GatedFusion(model.emb_size, 3).to(device)
    node_encoder = GatedFusion(model.emb_size, 2).to(device)

    max_dist = 500.0

    for i in range(len(pos_node_list)):
        pos_node_id = str(pos_node_list[i])
        poi_list = G_pos.nodes[pos_node_id]['poi_value']
        poi_id = G_pos.nodes[pos_node_id]['poi_id']

        if len(poi_id) != 0 and len(poi_id) <= 10:

            poi_feat_list = []

            for j in range(len(poi_id)):

                value, value_m = q_tensor_mask(poi_list[j])
                value = value.to(device)
                value_m = value_m.to(device)
                with torch.no_grad():
                    v_enc = model.lm(value, attention_mask=value_m)[0][:, :, :]
                    poi_emb = model.projection_p(torch.mean(torch.mean(v_enc, 0), 0))

                svi_memory = []
                distance_memory = []
                context = cd.poi[poi_id[j]]['svi']
                distance = cd.poi[poi_id[j]]['svi_dist']
                angle = cd.poi[poi_id[j]]['angle']

                with torch.no_grad():
                    for n in range(len(context)):
                        east_svi_emb, south_svi_emb, west_svi_emb, north_svi_emb = compute_svi_emb(
                            model, model.city, context[n])
                        svi_emb = compute_svi(
                            angle[n], east_svi_emb, south_svi_emb, west_svi_emb, north_svi_emb)

                        # del east_svi_emb, south_svi_emb, west_svi_emb, north_svi_emb
                        # torch.cuda.empty_cache()

                        if isinstance(svi_emb, tuple):
                            svi_emb = svi_emb[0]
                        svi_memory.append(svi_emb)
                        distance_memory.append(max_dist / (distance[n] + max_dist))

                svi_tensor = torch.stack(svi_memory).to(device)
                distance_tensor = torch.tensor(distance_memory).to(device)
                svi_emb = svi_generator(poi_emb, svi_tensor, distance_tensor)

                if cd.entities[poi_id[j]]['type'] == 'polygon':
                    raster = poly_to_raster(Polygon(cd.entities[poi_id[j]]['points']))
                    if not len(raster):
                        continue
                    with torch.no_grad():
                        image = torch.tensor(raster).unsqueeze(0).float().to(device)
                        image_emb = model.projection_img(model.cv(image).squeeze())
                        s = torch.tensor(norm_surface(cd.entities[poi_id[j]]['area'])).unsqueeze(-1).to(device)
                        s_emb = model.projection_s(s)

                    poly_emb = image_emb + s_emb / 2
                    poi_emb = poi_emb.unsqueeze(0)
                    svi_emb = svi_emb.unsqueeze(0)
                    poly_emb = poly_emb.unsqueeze(0)
                    poi_feat = polygon_encoder(poi_emb, svi_emb, poly_emb).squeeze()

                else:
                    poly_emb = torch.zeros(model.emb_size).to(device)
                    poi_emb = poi_emb.unsqueeze(0)
                    svi_emb = svi_emb.unsqueeze(0)
                    poly_emb = poly_emb.unsqueeze(0)
                    poi_feat = node_encoder(poi_emb, svi_emb, poly_emb).squeeze()

                poi_feat_list.append(poi_feat)

            if len(poi_feat_list) != 0:
                pos_value, _ = torch.max(torch.stack(poi_feat_list), 0)
                pos_value = torch.tensor(pos_value.tolist()).to(device)
                G_pos.nodes[pos_node_id]['value'] = pos_value
                
            else:
                G_pos.nodes[pos_node_id]['value'] = G_pos.nodes[pos_node_id]['value'].to(device)

        else:
            G_pos.nodes[pos_node_id]['value'] = G_pos.nodes[pos_node_id]['value'].to(device)

        if isinstance(G_pos.nodes[pos_node_list[i]]['value'], np.ndarray):
            G_pos.nodes[pos_node_list[i]]['value'] = torch.tensor(G_pos.nodes[pos_node_list[i]]['value'])

    data_pos = from_networkx(G_pos)
    return data_pos


class PolyContext(nn.Module):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    @staticmethod
    def forward(p0, p1, n0, n1):
        y = torch.tensor([1., 1., 0., 0.])
        p = torch.cat([p0, p1, n0, n1])

        loss = nn.BCELoss()
        return loss(p, y)


class InfoNCE(nn.Module):

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    @staticmethod
    def compute_batch_sim(a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, polyline_embs, c_embs):
        b_size = polyline_embs.shape[0]
        polyline_norm = F.normalize(polyline_embs, p=2, dim=1)
        c_norm = F.normalize(c_embs, p=2, dim=1)

        similarity_matrix = self.compute_batch_sim(polyline_norm, c_norm)

        sim_ij = torch.diag(similarity_matrix, b_size)
        sim_ji = torch.diag(similarity_matrix, -b_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)

        return loss


class RasterSim(nn.Module):

    def __init__(self, batch_size, hidden, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.hidden = hidden

    @staticmethod
    def compute_batch_sim(a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, t_embs, r_embs):
        loss = 0
        for i in range(self.batch_size):
            loss += torch.sum(torch.abs(t_embs[i] - r_embs[i]))

        loss = loss / self.batch_size / sqrt(self.hidden)

        return loss


class Semantic_Consistency(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, poi_emb, svi_memory):
        poi_emb = poi_emb.unsqueeze(0)

        poi_norm = F.normalize(poi_emb, p=2, dim=1)
        svi_norm = F.normalize(svi_memory, p=2, dim=1)

        similarities = F.cosine_similarity(poi_norm, svi_norm, dim=1)

        nominator = torch.exp(similarities / self.temperature)
        denominator = torch.sum(nominator)

        weights = nominator / denominator

        svi_emb = torch.matmul(weights, svi_memory)

        return svi_emb


class Spatial_Aware_Semantic_Consistency(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, poi_emb, svi_memory, distance_memory):
        poi_emb = poi_emb.unsqueeze(0)

        poi_norm = F.normalize(poi_emb, p=2, dim=1)
        svi_norm = F.normalize(svi_memory, p=2, dim=1)

        similarities = F.cosine_similarity(poi_norm, svi_norm, dim=1) * distance_memory
        # similarities = F.cosine_similarity(poi_norm, svi_norm, dim=1) / distance_memory

        nominator = torch.exp(similarities / self.temperature)
        denominator = torch.sum(nominator)

        weights = nominator / denominator

        svi_emb = torch.matmul(weights, svi_memory)

        return svi_emb


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores.maskd_fill_(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    res_scores = dropout(scores) if dropout is not None else scores

    return torch.matmul(res_scores, value), 0


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MHAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MHAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batches = query.size(0)

        Q, K, V = [linear(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                   for linear, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(Q, K, V, mask, self.dropout)

        x_all = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)

        return self.linears[-1](x_all)


def remove_random_poi_nodes(G):
    # num_nodes_to_remove = G.number_of_nodes() // 10
    # Identify all POI nodes in the graph
    poi_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'neighbour']
    poi_num = len(poi_nodes) // 10

    road_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'road']
    road_num = len(road_nodes) // 10

    # Randomly sample nodes to remove
    poi_remove = random.sample(poi_nodes, min(poi_num, len(poi_nodes)))
    road_remove = random.sample(road_nodes, min(road_num, len(road_nodes)))

    # Remove the selected nodes from the graph
    for node in poi_remove:
        G.remove_node(node)

    for node in road_remove:
        G.remove_node(node)

    return G


"""class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x"""

"""
def graph_to_data(G):
    edge_index = []
    edge_weight = []
    node_features = []

    for node in G.nodes(data=True):
        node_features.append(node[1]['value'])

    for edge in G.edges(data=True):
        edge_index.append([int(edge[0]), int(edge[1])])
        edge_weight.append(edge[2]['weight'])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # node_features = torch.stack(node_features, 0)
    node_features = torch.tensor(node_features, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
"""


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, task):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=True, dropout=0.6)
        self.task = task

    def forward(self, data, device=config.device):
        if self.task == 'poi_triplet':
            x, edge_index, edge_attr = data.value.to(device), data.edge_index.to(device), data.weight.to(device)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index, edge_attr)
        elif self.task == 'region_triplet':
            x, edge_index = data.value.to(device), data.edge_index.to(device)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
        return x


class GatedFusion(nn.Module):
    def __init__(self, emb_size, num):
        super(GatedFusion, self).__init__()
        self.hidden_size = emb_size
        self.dropout = 0.2
        self.num = num
        self.h = 32

        self.mh_attention = MHAttention(self.h, self.hidden_size, self.dropout)

        self.out_trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.poi_trans = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.svi_trans = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.poly_trans = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.gate = nn.Sequential(
            # nn.Linear(self.hidden_size * 2, self.hidden_size),
            # nn.ReLU(),
            # nn.Softmax(dim=0)
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

        self.filtration_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU()
        )

    def forward(self, poi_emb, svi_emb, poly_emb):

        poi_emb = self.poi_trans(poi_emb)
        svi_emb = self.svi_trans(svi_emb)

        svi_feat = self.mh_attention(svi_emb, poi_emb, poi_emb, None)
        poi_feat = self.mh_attention(poi_emb, svi_feat, svi_feat, None)

        poi_feat = poi_feat + poi_emb
        svi_feat = svi_feat + svi_emb

        # svi_attn_gate = self.gate(torch.cat([poi_feat.unsqueeze(0), svi_feat.unsqueeze(0)], dim=0)).squeeze()

        # svi_fusion = (poi_feat - svi_feat) * svi_attn_gate[0].unsqueeze(-1) + svi_feat
        svi_attn_gate = self.gate(torch.cat([poi_feat, svi_feat], dim=-1))  # [batch_size, embedding_dim*2]
        svi_fusion = svi_attn_gate * poi_feat + (1 - svi_attn_gate) * svi_feat

        out = self.out_trans(self.filtration_gate(torch.cat([poi_feat, svi_fusion], dim=-1)))

        if self.num == 3:

            poly_emb = self.poly_trans(poly_emb)
            poi_emb = out

            poly_feat = self.mh_attention(poly_emb, poi_emb, poi_emb, None)
            poi_feat = self.mh_attention(poi_emb, poly_feat, poly_feat, None)

            poly_feat = poly_feat + poly_emb
            poi_feat = poi_feat + poi_emb

            # poly_attn_gate = self.gate(torch.cat([poi_feat.unsqueeze(0), poly_feat.unsqueeze(0)], dim=0)).squeeze()

            # poly_fusion = (poi_feat - poly_feat) * poly_attn_gate[0].unsqueeze(-1) + poly_feat
            poly_attn_gate = self.gate(torch.cat([poi_feat, poly_feat], dim=-1))  # [batch_size, embedding_dim*2]
            poly_fusion = poly_attn_gate * poi_feat + (1 - poly_attn_gate) * poly_feat

            out = self.out_trans(self.filtration_gate(torch.cat([poi_feat, poly_fusion], dim=-1)))

        return out