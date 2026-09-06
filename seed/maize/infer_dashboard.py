import os, cv2, json, numpy as np
from .detect_seeds_shoots import detect_seeds, get_clean_shoot_mask
from .skeleton_graph import build_skeleton_graph, match_seeds_to_shoots

CLASS_COLORS = {
    'germinated': (0, 200, 0),
    'semi_germinated': (0, 200, 255),
    'non_germinated': (0, 0, 200),
}


def fallback_semigerminated(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    green=((h>=25)&(h<=55)&(s>=40)&(v>=60))
    return int(green.sum()) >= 2


def run(image_path, out_dir='output', gap_multiplier=2.0, downscale_width=1400):
    os.makedirs(out_dir, exist_ok=True)
    orig=cv2.imread(str(image_path))
    if orig is None: raise ValueError(f'Could not read image: {image_path}')
    h,w=orig.shape[:2]
    scale=min(1.0, downscale_width/w)
    img=cv2.resize(orig,(int(w*scale),int(h*scale))) if scale<1 else orig.copy()

    seeds=detect_seeds(img)
    shoot_mask=get_clean_shoot_mask(img)
    skel,G,degrees,endpoints=build_skeleton_graph(shoot_mask)
    matches=match_seeds_to_shoots(seeds,endpoints,G,degrees,gap_multiplier)

    tie=None
    try:
        from .tie_break_classify import TieBreakClassifier, extract_crop
        tie=TieBreakClassifier()
    except Exception as e:
        print('CLIP tie-break unavailable; using local green-growth fallback:', e)

    annotated=img.copy()
    counts={'germinated':0,'semi_germinated':0,'non_germinated':0}
    predictions=[]

    for seed,match in zip(seeds,matches):
        if match.get('shoot_found'):
            cls='germinated'; conf=None
            detail={'method':'skeleton_geometry','gap_to_endpoint':match.get('gap_to_endpoint'),
                    'shoot_path_length':match.get('shoot_path_length'),'termination':match.get('termination')}
        else:
            r=max(seed['w'],seed['h'])*5
            x0=max(0,seed['cx']-r); y0=max(0,seed['cy']-r)
            x1=min(img.shape[1],seed['cx']+r); y1=min(img.shape[0],seed['cy']+r)
            crop=img[y0:y1,x0:x1]
            if tie is not None:
                cls,conf=tie.classify_crop(crop)
                detail={'method':'clip_tie_break','gap_to_endpoint':match.get('gap_to_endpoint')}
            else:
                semi=fallback_semigerminated(crop)
                cls='semi_germinated' if semi else 'non_germinated'
                conf=None
                detail={'method':'local_green_fallback','gap_to_endpoint':match.get('gap_to_endpoint')}
        counts[cls]+=1
        x=seed['cx']-seed['w']//2; y=seed['cy']-seed['h']//2
        pred={'class_name':cls,'confidence':conf,'bbox_xywh':[x,y,seed['w'],seed['h']],'detail':detail}
        predictions.append(pred)
        color=CLASS_COLORS[cls]
        cv2.rectangle(annotated,(x,y),(x+seed['w'],y+seed['h']),color,2)
        cv2.putText(annotated,cls[:4],(x,max(0,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)

    total=sum(counts.values())
    strict=(counts['germinated']+counts['semi_germinated'])/total if total else 0
    weighted=(counts['germinated']+0.5*counts['semi_germinated'])/total if total else 0
    base=os.path.splitext(os.path.basename(str(image_path)))[0]
    ann_path=os.path.join(out_dir,f'{base}_annotated.jpg')
    json_path=os.path.join(out_dir,f'{base}_result.json')
    cv2.imwrite(ann_path,annotated)
    result={'crop':'Maize','image':str(image_path),'annotated_image':ann_path,
            'counts':counts,'total_seeds':total,'germinated_total':counts['germinated']+counts['semi_germinated'],
            'germination_rate':round(strict,4),'weighted_germination_rate':round(weighted,4),
            'predictions':predictions}
    with open(json_path,'w',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print(json.dumps(result['counts'],indent=2))
    return result
