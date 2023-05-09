import glob
import os
import shutil

import albumentations
import geopandas as gpd
import numpy as np
import psycopg2
import rasterio
import redis
import segmentation_models_pytorch as smp
import shapely
import torch
import torchvision
from eodag.api.core import EODataAccessGateway
from eoreader.bands import VV, VV_DSPK, VH, VH_DSPK
from eoreader.reader import Reader
from osgeo import gdal, osr
from rasterio.features import shapes
from sentinelhub.geo_utils import wgs84_to_utm, to_wgs84, get_utm_crs
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
# import requests
# import warnings
# import PIL.Image
# import torch.nn as nn
# from segmentation_models_pytorch.losses import DiceLoss
# from shapely import wkt, symmetric_difference

db = {
    "host": "127.0.0.1",
    "port": "5432",
    "user": "postgres",
    "password": "20010608Kd",
    "database": "django",
}
# workspace = 'C:/diploma/backend/deep_api/images/source'
# save_path = 'C:/diploma/backend/deep_api/images/input'
# out_path = 'C:/diploma/backend/deep_api/images/output'
# model_path = 'C:/diploma/backend/deep_api/models/ice.pth'
prefix = 'C:/diploma/backend/deep_api/'
sub_dir = ['images/source', 'images/input', 'images/output', 'models/ice.pth']
workspace, save_path, out_path, model_path = [prefix + i for i in sub_dir]
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "katarina.spasenovic@omni-energy.it"
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "M@rkon!1997"
yaml_content = """
peps:
    download:
        outputs_prefix: "{}"
        extract: true
""".format(workspace)

dag = EODataAccessGateway(f'{workspace}/eodag_conf.yml')
dag.set_preferred_provider("peps")


def convert(s):
    s = s % (24 * 3600)
    h = s // 3600
    s %= 3600
    m = s // 60
    s %= 60
    return "%02d:%02d:%02d" % (h, m, s)


def etl_cleanup():
    shutil.rmtree(workspace)

    for i in os.listdir(save_path):
        os.remove(f'{save_path}/{i}')

    for i in os.listdir(out_path):
        os.remove(f'{out_path}/{i}')

    if not os.path.isdir(workspace):
        os.mkdir(workspace)

    with open(f'{workspace}/eodag_conf.yml', "w") as f_yml:
        f_yml.write(yaml_content.strip())


class Order:
    def __init__(self, id_, owner, poly_wkt, imagery_start, imagery_end, crs, created_at, finished_at, predict):
        self.id = id_
        self.owner = owner
        self.poly_wkt = poly_wkt
        self.imagery_start = imagery_start
        self.imagery_end = imagery_end
        self.crs = crs
        self.created_at = created_at
        self.finished_at = finished_at
        self.predict = predict


def shorty(tiff_name, patch, ok_bands, geo_transform):
    nx, ny = patch.shape[1], patch.shape[1]
    np.save(tiff_name, patch)
    bd = 2 if len(ok_bands) == 2 else 3
    dst_ds = gdal.GetDriverByName('GTiff').Create(f'{tiff_name}.tiff', ny, nx, bd, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(geo_transform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(4326)  # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch[0])  # write r-band to the raster
    dst_ds.GetRasterBand(2).WriteArray(patch[1])  # write g-band to the raster
    if bd == 3:
        dst_ds.GetRasterBand(3).WriteArray(patch[2])  # write b-band to the raster
    dst_ds.FlushCache()


def eo_do_it(order: Order):
    dataset = gpd.GeoSeries.from_wkt(data=[order.poly_wkt], crs='epsg:4326').to_crs('epsg:4326')
    search_criteria = {
        "productType": 'S1_SAR_GRD',
        "start": f'{order.imagery_start}T00:00:00',
        "end": f'{order.imagery_end}T23:59:59',
        "geom": Polygon(dataset.iloc[0]),
        "items_per_page": 500,
    }

    first_page, estimated = dag.search(**search_criteria)
    if estimated == 0:
        print("no estimated")
        return

    print('download started, estimated =', estimated)
    for item in first_page:
        if {'1SDH', 'EW'} & set(item.properties["title"].split('_')):
            continue
        try:
            product_path = item.download(extract=False)
            break
        except Exception as e:
            print(e)
    print('download finished')

    try:
        zip_paths = [glob.glob(f'{workspace}/*1SDV*.zip')[0]]
    except IndexError:
        zip_paths = []
        print("no zips")

    for zip_id in trange(len(zip_paths), ascii=True):
        if not os.path.isfile(zip_paths[zip_id]):
            continue
        if order.id not in zip_paths[zip_id]:
            continue

        full_path = os.path.join(workspace, zip_paths[zip_id])
        try:
            product = Reader().open(full_path)
            print(product.resolution)
        except:
            continue

        product_poly = product.wgs84_extent().iloc[0].geometry
        name = os.path.basename(full_path)
        crs = get_utm_crs(product_poly.bounds[0], product_poly.bounds[1])

        poly = Polygon(unary_union(dataset.iloc[0]))
        inter_area = product_poly.intersection(poly).area
        if inter_area == 0:
            continue

        bands = [VV, VV_DSPK, VH, VH_DSPK]
        ok_bands = [band for band in bands if product.has_band(band)]
        if len(ok_bands) != 4:
            continue

        stack = product.stack(ok_bands)
        np_stack = stack.to_numpy()
        print(product.resolution)
        min_utm = wgs84_to_utm(poly.bounds[0], poly.bounds[1], crs)
        max_utm = wgs84_to_utm(poly.bounds[2], poly.bounds[3], crs)
        min_utm = max(min_utm[0], stack.x[0]), max(min_utm[1], stack.y[-1])
        max_utm = min(max_utm[0], stack.x[-1]), min(max_utm[1], stack.y[0])
        min_x = int((np.abs(stack.x - min_utm[0])).argmin())
        max_y = int((np.abs(stack.y - min_utm[1])).argmin())
        max_x = int((np.abs(stack.x - max_utm[0])).argmin())
        min_y = int((np.abs(stack.y - max_utm[1])).argmin())
        x_min, y_max = to_wgs84(stack.x[min_x], stack.y[min_y], crs)
        x_max, y_min = to_wgs84(stack.x[max_x], stack.y[max_y], crs)
        x_res, y_res = x_max - x_min, y_max - y_min
        geo_t = (x_min, x_res, 0, y_max, 0, -y_res)
        shorty(f'{save_path}/{order.id}', np_stack, ok_bands, geo_t)

        # resolution = product.resolution
        # chunk_size = int((256 / 20) * 100)
        #
        # min_utm = wgs84_to_utm(poly.bounds[0], poly.bounds[1], crs)
        # max_utm = wgs84_to_utm(poly.bounds[2], poly.bounds[3], crs)
        # min_utm = max(min_utm[0], stack.x[0]), max(min_utm[1], stack.y[-1])
        # max_utm = min(max_utm[0], stack.x[-1]), min(max_utm[1], stack.y[0])
        #
        # if min_utm[0] > max_utm[0] or min_utm[1] > max_utm[1]:
        #     continue
        #
        # min_x = int((np.abs(stack.x - min_utm[0])).argmin())
        # max_y = int((np.abs(stack.y - min_utm[1])).argmin())
        # max_x = int((np.abs(stack.x - max_utm[0])).argmin())
        # min_y = int((np.abs(stack.y - max_utm[1])).argmin())
        #
        # step_x = (max_x - min_x) // chunk_size
        # step_y = (max_y - min_y) // chunk_size
        # print(step_x, step_y)
        #
        # for sx in range(step_x + 1):
        #     for sy in range(step_y + 1):
        #         try:
        #             tiff_name = f'{save_path}/{order.id}_{sx}_{sy}'
        #             if tiff_name in os.listdir(save_path):
        #                 continue
        #
        #             y1 = min_y + sy * chunk_size
        #             y2 = min_y + (sy + 1) * chunk_size
        #
        #             x1 = min_x + sx * chunk_size
        #             x2 = min_x + (sx + 1) * chunk_size
        #             if sum([y1 < 0, x1 < 0, y2 > len(stack.y), x2 > len(stack.x)]):
        #                 continue
        #
        #             patch = np_stack[:, y1:y2, x1:x2]
        #             if np.sum(np.isnan(patch)) > 20 * chunk_size * 4:
        #                 if step_x != 0 and step_y != 0:
        #                     continue
        #
        #             x_min, y_max = to_wgs84(stack.x[x1], stack.y[y1], crs)
        #             x_max, y_min = to_wgs84(stack.x[x2], stack.y[y2], crs)
        #             nx, ny = patch.shape[1], patch.shape[1]
        #
        #             x_res, y_res = (x_max - x_min) / float(nx), (y_max - y_min) / float(ny)
        #             geo_transform = (x_min, x_res, 0, y_max, 0, -y_res)
        #
        #             np.save(tiff_name, patch)
        #
        #             bd = 2 if len(ok_bands) == 2 else 3
        #             dst_ds = gdal.GetDriverByName('GTiff').Create(f'{tiff_name}.tiff', ny, nx, bd, gdal.GDT_Byte)
        #             dst_ds.SetGeoTransform(geo_transform)  # specify coords
        #             srs = osr.SpatialReference()  # establish encoding
        #             srs.ImportFromEPSG(4326)  # WGS84 lat/long
        #             dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        #             dst_ds.GetRasterBand(1).WriteArray(patch[0])  # write r-band to the raster
        #             dst_ds.GetRasterBand(2).WriteArray(patch[1])  # write g-band to the raster
        #             if bd == 3:
        #                 dst_ds.GetRasterBand(3).WriteArray(patch[2])  # write b-band to the raster
        #             dst_ds.FlushCache()
        #         except Exception:
        #             print(f'FAIL at {sx}-{sy}', end='')
        #             pass


# MODEL RUNNERS
def item_getter(path: str, file_name: str, transforms=None):
    image = np.load(f'{path}/{file_name}', 'r')
    if transforms is None:
        transforms = albumentations.Compose([albumentations.CenterCrop(256, 256, p=1.0, always_apply=False)])
    # image = rasterio.open(f'{path}/{file_name}', 'r').read()

    img = image.transpose((1, 2, 0))
    augmented = transforms(image=img)
    image = albumentations.Compose([albumentations.Resize(256, 256, p=1.0, interpolation=3)])(image=augmented['image'])
    image = image['image'].transpose((2, 0, 1))
    assert not np.any(np.isnan(image))
    return image


class InferDataset(Dataset):
    def __init__(self, path, file_n):
        self.path = path
        self.data_list = [file_n]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        f_name = self.data_list[index]
        return item_getter(self.path, f_name)


def new_normalize(im: np.ndarray) -> np.ndarray:
    im = np.nan_to_num(im)
    # mean, std = np.zeros(im.shape[0]), np.zeros(im.shape[0])
    # for channel in range(im.shape[0]):
    #     mean[channel] = np.mean(im[channel, :, :])
    #     std[channel] = np.std(im[channel, :, :])

    mean = np.array([-16.388807, -16.38885, -30.692194, -30.692194])
    std = np.array([5.6070476, 5.6069245, 8.395209, 8.395208])
    mean[1], mean[3] = np.mean(im[1, :, :]), np.mean(im[3, :, :])
    std[1], std[3] = np.std(im[1, :, :]), np.std(im[3, :, :])

    # TODO try to change channels
    # mean[0], mean[2] = np.mean(im[0, :, :]), np.mean(im[2, :, :])
    # std[0], std[2] = np.std(im[0, :, :]), np.std(im[2, :, :])

    norm = torchvision.transforms.Normalize(mean, std)
    return np.asarray(norm.forward(torch.from_numpy(im)))


def coll_fn(batch_):
    ims_, labels_ = [], []
    for _, sample in enumerate(batch_):
        im = sample
        ims_.append(torch.from_numpy(im.copy()))
    return torch.stack(ims_, 0).type(torch.FloatTensor)


def to_pol(j):
    return shapely.Polygon([(k_[0], k_[1]) for k_ in [j for j in j['coordinates'][0]]])


def rasterio_geo(bytestream, mask_, key):
    geoms = []
    with rasterio.open(bytestream) as dataset:
        for geom, val in shapes(mask_, transform=dataset.transform):
            if val == key:
                g = rasterio.warp.transform_geom(dataset.crs, 'EPSG:3857', geom)
                geoms.append(g)
            # geoms[val].append(rasterio.warp.transform_geom(dataset.crs, 'EPSG:3857', geom))  # , precision=6
    return geoms


def process_image_ice(src_path, name_, output_img_path):
    transform = albumentations.Compose([albumentations.Resize(256, 256, p=1.0, interpolation=3)])
    img = np.load(f'{src_path}/{name_}.npy')
    img = np.asarray(img).transpose((1, 2, 0))
    img = transform(image=img)['image']
    img = np.transpose(img, (2, 0, 1))

    # img = stand(img, single_stand=True)
    img = new_normalize(img)
    to_pil = (img[:3].transpose((1, 2, 0))).astype(np.uint8)
    # PIL.Image.fromarray(to_pil).show()

    model = smp.DeepLabV3(
        encoder_name="timm-mobilenetv3_small_075",
        encoder_weights=None,
        in_channels=4,
        classes=6,
    ).to(torch.device('cpu'), dtype=torch.float32)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict']
    model.load_state_dict(state_dict)

    dataset = InferDataset(src_path, name_ + '.npy')
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=coll_fn, shuffle=False)

    inputs = next(iter(dataloader))
    inputs = inputs.to('cpu')
    model.eval()
    outputs = model(inputs)
    y_pred = np.array(torch.argmax(outputs, dim=1).cpu())

    palette0 = np.array([
        [0, 0, 255],
        [255, 255, 0],
        [255, 128, 0],
        [255, 0, 0],
        [0, 128, 0],
    ])
    pred_to_pil = palette0[y_pred[0].astype(np.uint8)].astype(np.uint8)
    # PIL.Image.fromarray(pred_to_pil).show()

    profile = rasterio.open(f'{src_path}/{name_}.tiff', 'r').profile
    profile["count"] = 1
    with rasterio.open(f'{output_img_path}/{name_}.tiff', 'w', **profile) as src:
        src.write(y_pred[0].astype(np.uint8), 1)
    return f'{output_img_path}/{name_}.tiff'


# DATABASE
def redis_connect():
    try:
        return redis.Redis(host="127.0.0.1", port=6379)
    except Exception as e:
        print("REDIS CONNECT EXCEPTION", e)


def redis_get_all():
    order_lst = []
    try:
        order_lst = redis_connect().lrange("orderList", start=0, end=-1)
    except Exception as e:
        print("REDIS GET EXCEPTION", e)

    return [order_id.decode("utf-8") for order_id in order_lst]


def redis_pop(order_id: str):
    try:
        redis_connect().lrem("orderList", count=0, value=order_id)
    except Exception as e:
        print("REDIS POP EXCEPTION", e)


def get_order(order_id):
    connection = cursor = None
    try:
        connection = psycopg2.connect(
            user=db["user"],
            password=db["password"],
            host=db["host"],
            port=db["port"],
            database=db["database"]
        )
        cursor = connection.cursor()
        q = f"""
        SELECT id, "owner", poly_wkt, imagery_start, imagery_end, crs, created_at, finished_at, predict
        FROM public.showcase_order
        WHERE id='{order_id}'::uuid;
        """
        cursor.execute(q)
        return Order(*cursor.fetchall()[0])

    except (Exception, psycopg2.Error) as error:
        print("Failed to get orders", error)

    finally:
        if connection is not None and connection:
            cursor.close()
            connection.close()


def save_predict(order):
    connection = cursor = None
    try:
        connection = psycopg2.connect(
            user=db["user"],
            password=db["password"],
            host=db["host"],
            port=db["port"],
            database=db["database"]
        )
        cursor = connection.cursor()

        q = f"""
        UPDATE public.showcase_order
        SET finished_at='{order.finished_at}', predict='{order.predict}'
        WHERE id='{order.id}'::uuid;
        """
        cursor.execute(q)
        connection.commit()

    except (Exception, psycopg2.Error) as error:
        print("Failed to insert record into showcase_order", error)

    finally:
        if connection is not None and connection:
            cursor.close()
            connection.close()
