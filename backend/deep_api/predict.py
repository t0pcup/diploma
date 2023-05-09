from helpers import *


def save_order_result_to_database(db, order):
    try:
        connection = psycopg2.connect(
            user=db["user"],
            password=db["password"],
            host=db["host"],
            port=db["port"],
            database=db["database"]
        )
        cursor = connection.cursor()

        q = """UPDATE orders
                SET status = %s, url = %s, url2 = %s, finished_at = %s, result = %s, result2 = %s, bbox = %s, diff = %s
                WHERE id = %s"""
        record = (
            order["status"],
            order["url"],
            order["url2"],
            datetime.datetime.now(),
            order["result"],
            order["result2"],
            order["bbox"],
            order["diff"],
            order["order_id"],
        )
        cursor.execute(q, record)
        connection.commit()

    except (Exception, psycopg2.Error) as error:
        print("Failed to insert record into mobile table", error)

    finally:
        # closing database connection.
        if connection is not None and connection:
            cursor.close()
            connection.close()

def item_getter(path: str, file_name: str, transforms=A.Compose([
    A.CenterCrop(256, 256, p=1.0, always_apply=False)]), val=False):
    image = np.load(f'{path}/{file_name}', 'r')
    # image = rasterio.open(f'{path}/{file_name}', 'r').read()

    img = image.transpose((1, 2, 0))
    augmented = transforms(image=img)
    image = A.Compose([A.Resize(256, 256, p=1.0, interpolation=3)])(image=augmented['image'])['image']
    image = image.transpose((2, 0, 1))
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
        image = item_getter(self.path, f_name, val=False)
        return image


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


def process_image_ice(src_path, name_, model_path, output_img_path):
    transform = A.Compose([A.Resize(256, 256, p=1.0, interpolation=3)])
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
    ).to('cpu', dtype=torch.float32)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict']
    model.load_state_dict(state_dict)

    dataset = InferDataset(src_path, name_ + '.npy')
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=coll_fn, shuffle=False)

    inputs = next(iter(dataloader))
    inputs = inputs.to('cpu')
    model.eval()
    outputs = model(inputs)
    y_pred = np.array(torch.argmax(outputs, dim=1).cpu())

    # y_pred[y_pred < 2] = 0.0
    # # y_pred[y_pred == 5] = 0
    # y_pred[y_pred != 0] = 1.0
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


def t(lst: list):
    return shapely.Polygon([(k_[0], k_[1]) for k_ in [j for j in lst[0]]])


def to_pol(j):
    return t(j['coordinates'])


def rasterio_geo(bytestream, mask_, key):
    geoms = []
    with rasterio.open(bytestream) as dataset:
        for geom, val in shapes(mask_, transform=dataset.transform):
            if val == key:
                # print(val)
                g = rasterio.warp.transform_geom(dataset.crs, 'EPSG:3857', geom)
                geoms.append(g)
            # geoms[val].append(rasterio.warp.transform_geom(dataset.crs, 'EPSG:3857', geom))  # , precision=6
    return geoms


order_id = sys.argv[1]
url = sys.argv[2].replace('jpeg', 'tiff')
save_path = 'C:/diploma/sip-service-main/sip-service-auth/src/main/python/images/input'
out_path = 'C:/diploma/sip-service-main/sip-service-auth/src/main/python/images/output'
model_p = 'C:/diploma/sip-service-main/sip-service-auth/src/main/python/models/ice.pth'
res_pred = [[] for i in range(6)]

for i in os.listdir(out_path):
    os.remove(f'{out_path}/{i}')

for i in os.listdir(save_path):
    if '.npy' in i:
        continue
    name = i.split('.')[0]

    predict_name = process_image_ice(save_path, name, model_p, out_path)
    image = rasterio.open(predict_name, 'r')
    im_arr = np.asarray(image.read()).transpose((1, 2, 0))
    mask = im_arr.reshape(im_arr.shape[:2])
    h, w = mask.shape
    mask_cp = np.zeros((h, w), dtype='uint8')
    for k in np.unique(mask):
        mask_cp[mask[:, :] == k] = 255
        mask_cp[mask[:, :] != k] = 0
        res = rasterio_geo(predict_name, mask, float(k))
        for p in res:
            res_pred[k].append(to_pol(p))
        # print(k, res_pred, len(res))

result = ''
bb_mp = []
for k in range(len(res_pred)):
    if len(res_pred[k]) != 0:
        mp = shapely.MultiPolygon(res_pred[k])
        mp = shapely.unary_union(mp)
        bb_mp.append(mp)
        gpd.GeoSeries(mp).to_file(f"{out_path}/{k}.json", driver='GeoJSON', show_bbox=False)
        with open(f"{out_path}/{k}.json", mode='r') as j_source:
            result += j_source.read().replace('\n', '') + '\n'
    else:
        result += '\n'

bbox = gpd.GeoSeries(shapely.unary_union(bb_mp)).to_json().split("bbox\": [")[1].split("]")[0]

# print(result)
db = {
    "host": "127.0.0.1",
    "port": "5432",
    "user": "postgres",
    "password": "20010608Kd",
    "database": "db",
}
order = {
    "status": "true",
    "url": url,
    "url2": None,
    "bbox": bbox,
    "result": result,
    "result2": None,
    "order_id": order_id,
    "diff": None,
}
save_order_result_to_database(db, order)
