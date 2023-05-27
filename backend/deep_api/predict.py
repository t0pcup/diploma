import os
import uuid
import random
from datetime import datetime, timedelta

from helpers import *

# get_predict(get_order(sys.argv[1]))
d_lst, lst = [], []
green_lst = os.listdir('C:/a/data10-5')
path_ = 'D:/AURORA/NorthenSeaRoute/ICE/stage_4/dafuck/ds/data/image/npy'
path_new = 'D:/AURORA/NorthenSeaRoute/ICE/stage_4/dafuck/ds/data/image'


class Orda:
    def __init__(self, date, time, shape, pred):
        y = date[:4]
        m = date[4:6]
        d = date[6:]
        self.date = f'{y}-{m}-{d}'
        self.time = time
        self.shape = shape
        self.pred = pred

    def __str__(self):
        return '\t'.join([self.date, self.time, self.shape, self.pred])

    def save(self):
        with open(f'{path_new}/{self.date}_{self.time}_shape.txt', mode='w') as f:
            f.write(self.shape)
        with open(f'{path_new}/{self.date}_{self.time}_pred.txt', mode='w') as f:
            f.write(self.pred)


def dum(o_: Orda, st_):
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
        end_ = st_ + timedelta(minutes=38) + timedelta(seconds=random.randint(0, 100))

        q = f"""
        INSERT INTO public.showcase_order
        (id, "owner", poly_wkt, imagery_start, imagery_end, crs, created_at, finished_at, predict)
        VALUES('{uuid.uuid1()}'::uuid, '8bdfe560-f40c-11ed-9926-00583f12d837', '{o_.shape}', '{o_.date}', '{o_.date}', 'EPSG:3857', '{st_}', '{end_}', '{o_.pred}');
        """
        cursor.execute(q)
        connection.commit()

    except (Exception, psycopg2.Error) as error:
        print("Failed to insert record into showcase_order", error)

    finally:
        if connection is not None and connection:
            cursor.close()
            connection.close()


gener = []
shuf = np.random.permutation(os.listdir(path_))
cnt = 0
for i in shuf:
    if '.npy' not in i:
        if i.replace('.tiff', '.npy') not in os.listdir(path_):
            os.remove(f'{path_}/{i}')
            print('del')
        continue

    try:
        cnt += 1
        start = datetime.datetime.now() - timedelta(days=int(np.random.choice([2, 3, 4, 5, 6, 7])))
        p, kkk_ = tester(i)
        t_i = i.replace('.npy', '.tiff')
        ds = gdal.Open(f'{path_}/{t_i}')
        w, h = ds.RasterXSize, ds.RasterYSize
        gt = ds.GetGeoTransform()
        minx, maxy = gt[0], gt[3]
        miny = gt[3] + w * gt[4] + h * gt[5]
        maxx = gt[0] + w * gt[1] + h * gt[2]
        d = '2021' + i.split('_')[1]

        o = Orda(date=d.split('T')[0],
                 time=d.split('T')[1],
                 shape=Polygon([
                     [minx, miny],
                     [maxx, miny],
                     [maxx, maxy],
                     [minx, maxy],
                     [minx, miny],
                 ]).wkt,
                 pred=p)
        gener.append(o)
        # o.save()
        dum(o, start)

    except Exception as e:
        print(e)
        _ = 0

    if cnt == 1:
        break

print(len(gener))
