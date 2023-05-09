import psycopg2
import redis
import glob
import os
import shutil

from eodag import setup_logging
from eodag.api.core import EODataAccessGateway
from eoreader.bands import VV, VV_DSPK, VH, VH_DSPK
from eoreader.reader import Reader
from osgeo import gdal, osr
from sentinelhub.geo_utils import wgs84_to_utm, to_wgs84, get_utm_crs
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import trange
import os
import torch
import shapely
import datetime
import rasterio
import requests
import psycopg2
import warnings
import PIL.Image
import sys
import numpy as np
import torch.nn as nn
import geopandas as gpd
import albumentations as A
import segmentation_models_pytorch as smp
import torchvision

from rasterio.features import shapes
from shapely import wkt, symmetric_difference
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.losses import DiceLoss

db = {
    "host": "127.0.0.1",
    "port": "5432",
    "user": "postgres",
    "password": "20010608Kd",
    "database": "django",
}
workspace = 'C:/diploma/backend/deep_api/images/source'
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "katarina.spasenovic@omni-energy.it"
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "M@rkon!1997"
save_path = 'C:/diploma/backend/deep_api/images/input'
yaml_content = """
peps:
    download:
        outputs_prefix: "{}"
        extract: true
""".format(workspace)


dag = EODataAccessGateway(f'{workspace}/eodag_conf.yml')
dag.set_preferred_provider("peps")


def etl_cleanup():
    shutil.rmtree(workspace)
    if not os.path.isdir(workspace):
        os.mkdir(workspace)

    with open(f'{workspace}/eodag_conf.yml', "w") as f_yml:
        f_yml.write(yaml_content.strip())

    for i in os.listdir(save_path):
        os.remove(f'{save_path}/{i}')


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


def redis_get_all():
    order_lst = []
    try:
        r_server = redis.Redis(host="127.0.0.1", port=6379)
        order_lst = r_server.lrange("orderList", start=0, end=-1)
    except Exception as e:
        print("REDIS EXCEPTION", e)

    return [order_id.decode("utf-8") for order_id in order_lst]


def redis_pop(order_id: str):
    try:
        r_server = redis.Redis(host="127.0.0.1", port=6379)
        r_server.lrem("orderList", count=0, value=order_id)
    except Exception as e:
        print("REDIS EXCEPTION", e)


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
        res = cursor.fetchall()
        connection.commit()
        return Order(*res[0])

    except (Exception, psycopg2.Error) as error:
        print("Failed to get orders", error)

    finally:
        if connection is not None and connection:
            cursor.close()
            connection.close()


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

    print('download started, estimated=', estimated)
    for item in first_page:
        if {'1SDH', 'EW'} & set(item.properties["title"].split('_')):
            continue
        try:
            print(item.properties["title"].split('_'))
            product_path = item.download(extract=False)
            print(product_path)
            break
        except Exception as e:
            print(e)
            pass
    print('download finished')

    try:
        zip_paths = [glob.glob(f'{workspace}/*1SDV*.zip')[0]]
    except IndexError:
        zip_paths = []
        print("no zips")

    for zip_id in trange(len(zip_paths), ascii=True):
        if not os.path.isfile(zip_paths[zip_id]):
            continue
        full_path = os.path.join(workspace, zip_paths[zip_id])
        reader = Reader()
        try:
            product = reader.open(full_path)
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
        resolution = product.resolution
        chunk_size = int((256 / 20) * 100)

        min_utm = wgs84_to_utm(poly.bounds[0], poly.bounds[1], crs)
        max_utm = wgs84_to_utm(poly.bounds[2], poly.bounds[3], crs)
        min_utm = max(min_utm[0], stack.x[0]), max(min_utm[1], stack.y[-1])
        max_utm = min(max_utm[0], stack.x[-1]), min(max_utm[1], stack.y[0])

        if min_utm[0] > max_utm[0] or min_utm[1] > max_utm[1]:
            continue

        min_x = int((np.abs(stack.x - min_utm[0])).argmin())
        max_y = int((np.abs(stack.y - min_utm[1])).argmin())
        max_x = int((np.abs(stack.x - max_utm[0])).argmin())
        min_y = int((np.abs(stack.y - max_utm[1])).argmin())

        step_x = (max_x - min_x) // chunk_size
        step_y = (max_y - min_y) // chunk_size
        for sx in range(step_x + 1):
            for sy in range(step_y + 1):
                try:
                    tiff_name = f'{save_path}/{order.id}_{sx}_{sy}'
                    if tiff_name in os.listdir(save_path):
                        continue

                    y1 = min_y + sy * chunk_size
                    y2 = min_y + (sy + 1) * chunk_size

                    x1 = min_x + sx * chunk_size
                    x2 = min_x + (sx + 1) * chunk_size
                    if sum([y1 < 0, x1 < 0, y2 >= len(stack.y), x2 >= len(stack.x)]):
                        continue

                    patch = np_stack[:, y1:y2, x1:x2]
                    if np.sum(np.isnan(patch)) > 20 * chunk_size * 4:
                        continue

                    x_min, y_max = to_wgs84(stack.x[x1], stack.y[y1], crs)
                    x_max, y_min = to_wgs84(stack.x[x2], stack.y[y2], crs)
                    nx, ny = patch.shape[1], patch.shape[1]

                    x_res, y_res = (x_max - x_min) / float(nx), (y_max - y_min) / float(ny)
                    geo_transform = (x_min, x_res, 0, y_max, 0, -y_res)

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
                except:
                    print(f'FAIL at {sx}-{sy}', end='')
                    pass


