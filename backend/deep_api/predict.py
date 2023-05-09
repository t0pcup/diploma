import datetime
import sys

from helpers import *

order = get_order(sys.argv[1])
res_predict = [[] for i in range(6)]

for i in os.listdir(save_path):
    if '.npy' in i or order.id not in i:
        continue
    name = i.split('.')[0]

    predict_name = process_image_ice(save_path, name, out_path)
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
            res_predict[k].append(to_pol(p))
        # print(k, res_predict, len(res))

result = ''
for k in range(len(res_predict)):
    if len(res_predict[k]) != 0:
        mp = shapely.MultiPolygon(res_predict[k])
        mp = shapely.unary_union(mp)
        gpd.GeoSeries(mp).to_file(f"{out_path}/{k}.json", driver='GeoJSON', show_bbox=False)
        with open(f"{out_path}/{k}.json", mode='r') as j_source:
            result += j_source.read().replace('\n', '') + '\n'
    else:
        result += '\n'

order.predict = result
order.finished_at = datetime.datetime.now()
# bbox = gpd.GeoSeries(shapely.unary_union(bb_mp)).to_json().split("bbox\": [")[1].split("]")[0]
save_predict(order)
redis_pop(order.id)
