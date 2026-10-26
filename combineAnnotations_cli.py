import argparse
import io
import json
import numpy as np
from collections import defaultdict
from pycocotools import mask as maskUtils
from multiprocessing import Pool
from itertools import chain

annsDict = defaultdict(list)
imgDict = {}
maxIoU = 0.2

def annToRLE(ann):
    h, w = imgDict[ann['image_id']]
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def processImage(imgID):
    anns = annsDict[imgID]
    anns.sort(key=lambda item:item['score'])
    anns = anns[::-1]
    finalAnns = []
    for ann1 in anns:
        maxIoU_1_2 = 0
        if ann1['score']==2:
            if not ann1 in finalAnns:
                finalAnns.append(ann1)
            continue
        for ann2 in anns:
            if ann2['score']==2:
                continue
            if ann1 == ann2:
                continue
            if ann2 in finalAnns:
                iou = maskUtils.iou([ann1['rle']], [ann2['rle']], [False])[0]
                maxIoU_1_2=max(maxIoU_1_2,iou)
                if maxIoU_1_2 > maxIoU:
                    break
        if maxIoU_1_2< maxIoU:
            if not ann1 in finalAnns:
                finalAnns.append(ann1)

    return finalAnns


def parseArgs():
    parser = argparse.ArgumentParser(description='Combine mask, box, point and image annotations by annotation type.')
    parser.add_argument('mask', help='JSON file with mask annotations')
    parser.add_argument('box', help='JSON file with SAM box annotations')
    parser.add_argument('pts', help='JSON file with SAM point annotations')
    parser.add_argument('img', help='JSON file with SAM image annotations')
    parser.add_argument('output', help='output JSON file for the combined annotations')
    parser.add_argument('--max-iou', type=float, default=0.2, help='maximum IoU with already selected annotations (default: 0.2)')
    parser.add_argument('--processes', type=int, default=4, help='number of worker processes (default: 4)')
    return parser.parse_args()


def main():
    global maxIoU
    args = parseArgs()
    maxIoU = args.max_iou

    with open(args.mask) as f:
        data_mask=json.load(f)

    with open(args.box) as f:
        data_box=json.load(f)

    with open(args.pts) as f:
        data_pts=json.load(f)

    with open(args.img) as f:
        data_img=json.load(f)

    for img in data_mask['images']:
        imgDict[img['id']]=((img['height'], img['width']))

    for ann in data_mask['annotations']:
        ann['score']=np.float64(4)
        ann['rle']=annToRLE(ann)
        annsDict[ann['image_id']].append(ann)

    for ann in data_box['annotations']:
        ann['score']=np.float64(ann['SAM_score']+2)
        ann['rle']=annToRLE(ann)
        annsDict[ann['image_id']].append(ann)

    for ann in data_pts['annotations']:
        ann['score']=np.float64(ann['SAM_score']+1)
        ann['rle']=annToRLE(ann)
        annsDict[ann['image_id']].append(ann)

    for ann in data_img['annotations']:
        ann['score']=np.float64(ann['SAM_score'])
        ann['rle']=annToRLE(ann)
        annsDict[ann['image_id']].append(ann)

    with Pool(args.processes) as p:
        newAnns = list(p.map(processImage,imgDict.keys()))

    annID=1
    finalAnns = []
    for newAnn in chain(*newAnns):
        newAnn['segmentation']=newAnn['rle']
        newAnn.pop('rle')
        newAnn['id']=int(annID)
        finalAnns.append(newAnn)
        annID+=1

    data_mask['annotations']=finalAnns

    with io.open(args.output, 'w', encoding='utf-8') as f:
        str_ = json.dumps(data_mask, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        f.write(str_)


if __name__ == '__main__':
    main()
