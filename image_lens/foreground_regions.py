"""One frozen local foreground candidate. No future references, NN coordinates or semantic labels."""
import math
import hashlib
PARAMS=dict(padding_px=12,iterations=3,rng_seed=0,maximum_area_ratio=3.0,inner_border_px=1,background_delta=2,perimeter_consensus=.90,frame_background_fraction=.50)
def background_model(rgb):
 import numpy as np
 sides=[rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]]
 ring=np.concatenate(sides)
 colors,counts=np.unique(ring,axis=0,return_counts=True)
 color=colors[int(np.argmax(counts))]
 close=lambda data:np.max(np.abs(data.astype(np.int16)-color.astype(np.int16)),axis=-1)<=PARAMS['background_delta']
 side_fractions=[float(close(s).mean())for s in sides]
 fraction=float(close(rgb).mean())
 return dict(color_rgb=[int(x)for x in color],side_fractions=side_fractions,frame_fraction=fraction,
             available=min(side_fractions)>=PARAMS['perimeter_consensus']and fraction>=PARAMS['frame_background_fraction'])

def refine(rgb, original_mask):
 import cv2
 import numpy as np
 if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or original_mask.shape!=rgb.shape[:2] or original_mask.dtype!=np.bool_:
  raise ValueError('native RGB8 and same-size boolean support required')
 ys,xs=np.nonzero(original_mask)
 if len(xs)<8:return dict(status='unknown',reason='insufficient_seed'),None
 h,w=original_mask.shape;pad=PARAMS['padding_px'];x0=max(0,int(xs.min())-pad);y0=max(0,int(ys.min())-pad);x1=min(w,int(xs.max())+1+pad);y1=min(h,int(ys.max())+1+pad)
 bg_model=background_model(rgb)
 if not bg_model['available']:return dict(status='unknown',reason='flat_background_not_supported',background=bg_model),None
 color=np.asarray(bg_model['color_rgb'],np.int16)
 bg_match=np.max(np.abs(rgb[y0:y1,x0:x1].astype(np.int16)-color),axis=2)<=PARAMS['background_delta']
 old_seed=original_mask[y0:y1,x0:x1];seed=old_seed&~bg_match
 removed=int(old_seed.sum()-seed.sum())
 if seed.sum()<8:return dict(status='unknown',reason='foreground_background_seed_conflict',background=bg_model),None
 mask=np.full(seed.shape,cv2.GC_PR_BGD,dtype=np.uint8)
 ax,ay=int(xs.min())-x0,int(ys.min())-y0;bx,by=int(xs.max())+1-x0,int(ys.max())+1-y0
 mask[ay:by,ax:bx]=cv2.GC_PR_FGD;mask[bg_match]=cv2.GC_BGD;mask[seed]=cv2.GC_FGD
 if seed[0].any()or seed[-1].any()or seed[:,0].any()or seed[:,-1].any():return dict(status='unknown',reason='seed_at_image_boundary'),None
 mask[0]=mask[-1]=cv2.GC_BGD;mask[:,0]=mask[:,-1]=cv2.GC_BGD
 bg,fg=np.zeros((1,65),np.float64),np.zeros((1,65),np.float64)
 cv2.setRNGSeed(PARAMS['rng_seed'])
 cv2.grabCut(np.ascontiguousarray(rgb[y0:y1,x0:x1,::-1]),mask,None,bg,fg,PARAMS['iterations'],cv2.GC_INIT_WITH_MASK)
 foreground=(mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)
 _,labels=cv2.connectedComponents(foreground.astype(np.uint8),connectivity=8)
 ids=np.unique(labels[seed]);ids=ids[ids!=0];selected=np.isin(labels,ids);yy,xx=np.nonzero(selected)
 if not len(xx):return dict(status='unknown',reason='no_attached_foreground'),None
 box=[int(xx.min())+x0,int(yy.min())+y0,int(xx.max())+1+x0,int(yy.max())+1+y0]
 area=int(selected.sum());reason=None
 if selected[1].any()or selected[-2].any()or selected[:,1].any()or selected[:,-2].any():reason='foreground_at_crop_constraint'
 elif area>PARAMS['maximum_area_ratio']*len(xs):reason='foreground_area_expanded'
 return dict(status='unknown'if reason else'foreground_hypothesis',reason=reason,bbox=box if not reason else None,
             diagnostic_bbox=box,roi=[x0,y0,x1,y1],area_px=area,seed_area_px=int(len(xs)),background=bg_model,removed_background_seed_pixels=removed,
             retained_seed_roi_sha256=hashlib.sha256(seed.astype(np.uint8).tobytes()).hexdigest(),
             background_color_assumption_verified=False,
             full_part_verified=False,identity_verified=False),selected


def inspect(packet, report, region_api, decoded_frames):
 """Sparse diagnostic only. Original packet, centroids, trajectory and association are immutable."""
 import numpy as np
 import cv2
 cv2.setNumThreads(1)
 data=packet['region_observations'];profiles=data.get('profiles',{})
 bound={f['point_id'] for f in report['features']};table={f['frame']:{r['region_id']:r for r in f['regions']}for f in data['frames']}
 output=[];masks={}
 for n,rgb in decoded_frames:
  rows=[];hsv,lab=region_api._converted(rgb)
  for pid in sorted(bound):
   source=table[n][pid];common=dict(region_id=pid,identity_verified=False,full_part_verified=False,used_for_motion=False)
   if source['status']!='region_hypothesis':
    rows.append(dict(common,status='unknown',reason='original_'+source['status'],bbox=None));continue
   box=source['bbox'];area=(min(rgb.shape[1],box[2]+12)-max(0,box[0]-12))*(min(rgb.shape[0],box[3]+12)-max(0,box[1]-12))
   if area>262144:rows.append(dict(common,status='unknown',reason='inspection_roi_too_large',bbox=None));continue
   profile=profiles.get(pid)
   if (type(profile)is not dict or any(type(profile.get(k))is not list or len(profile[k])!=3 or any(type(v)is not int or not 0<=v<=255 for v in profile[k])for k in ['hsv','lab'])or profile['hsv'][0]>179):
    raise ValueError('bounded stored color profile required for foreground inspection')
   labels,stats,centers=region_api._components(hsv,lab,profile)
   matches=[i for i in range(1,len(stats))if [int(stats[i,0]),int(stats[i,1]),int(stats[i,0]+stats[i,2]),int(stats[i,1]+stats[i,3])]==box and int(stats[i,4])==source['area_px']and math.dist(centers[i],source['centroid'])<1e-5]
   if len(matches)!=1:rows.append(dict(common,status='unknown',reason='stored_support_not_reproduced',bbox=None));continue
   seed=labels==matches[0];result,mask=refine(rgb,seed);result.update(common,original_seed_sha256=hashlib.sha256(seed.astype(np.uint8).tobytes()).hexdigest())
   if result['status']=='foreground_hypothesis':masks[n,pid]=dict(roi=result['roi'],bbox=result['bbox'],mask=mask)
   rows.append(result)
  output.append(dict(frame=n,regions=rows))
 count=sum(len(f['regions'])for f in output);available=len(masks)
 return dict(schema='team-eye.foreground-inspection/1',
             state='unavailable'if not available else'partial'if available<count else'hypotheses_available',
             available_regions=available,unknown_regions=count-available,
             scope='Selected source frames only; not a tracked whole-part mask or anatomical/identity/ease truth.',
             background_exclusion_assumption='Pixels matching a sufficiently flat frame-border color are excluded; matching-color paint may be lost.',
             full_part_verified=False,used_for_motion=False,frames=output),masks


def markdown(inspection):
 import json
 lines=['','## 평탄 배경의 윤곽 검수','배경색과 같은 픽셀을 제외한 가설이다. 같은 색으로 칠한 물체 부분도 빠질 수 있다. 전체 부위·관절·이동 궤적의 정답으로 쓰지 않는다.']
 for frame in inspection['frames']:
  for row in frame['regions']:
   label=json.dumps(row['region_id'],ensure_ascii=True)
   lines.append(f"- f{frame['frame']} / {label}: {row['status']}; bbox={row.get('bbox')}; reason={row.get('reason')}")
 return '\n'.join(lines)+'\n'
