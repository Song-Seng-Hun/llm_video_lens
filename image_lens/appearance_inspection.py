"""Multi-warp RGB explanation; source appearance evidence, never anatomy or identity."""
import math
PARAMS=dict(max_side=480,nfeatures=1200,edge=15,fast=12,neighbors=4,hamming=64,near_best=16,
            models=6,min_inliers=16,min_span_x=.06,min_span_y=.12,margin=12,
            scale_min=.75,scale_max=1.33,angle=25.,translation_fraction=.65,
            ransac_px=2.5,iterations=2000,confidence=.99,blur=3,
            translation_scale_deviation=.04,translation_angle=3.,translation_support=.8,
            static_delta=1.5,residual=24.,min_features=32)

def thumbnail(bgr):
    import cv2
    h,w=bgr.shape[:2];s=min(1.,PARAMS['max_side']/max(h,w))
    return cv2.resize(bgr,(max(2,round(w*s)),max(2,round(h*s))),interpolation=cv2.INTER_AREA)

def hull_mask(points,shape):
    import cv2
    import numpy as np
    mask=np.zeros(shape,np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(points).astype(np.int32)),1)
    k=2*PARAMS['margin']+1
    return cv2.dilate(mask,np.ones((k,k),np.uint8)).astype(bool)

def models(a,b):
    import cv2
    import numpy as np
    orb=cv2.ORB_create(nfeatures=PARAMS['nfeatures'],edgeThreshold=PARAMS['edge'],fastThreshold=PARAMS['fast'])
    ka,da=orb.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None);kb,db=orb.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
    counts=[len(ka),len(kb)]
    if min(counts)<PARAMS['min_features'] or da is None or db is None:return [],counts
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING);pairs=set()
    for reverse,(x,y)in enumerate([(da,db),(db,da)]):
        for matches in matcher.knnMatch(x,y,k=min(PARAMS['neighbors'],len(y))):
            if not matches:continue
            cap=min(PARAMS['hamming'],matches[0].distance+PARAMS['near_best'])
            for m in matches:
                if m.distance<=cap:pairs.add((m.trainIdx,m.queryIdx)if reverse else(m.queryIdx,m.trainIdx))
    pairs=sorted(pairs);result=[]
    for _ in range(PARAMS['models']):
        if len(pairs)<PARAMS['min_inliers']:break
        x=np.float32([ka[i].pt for i,j in pairs]);y=np.float32([kb[j].pt for i,j in pairs]);cv2.setRNGSeed(0)
        matrix,mask=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=PARAMS['ransac_px'],maxIters=PARAMS['iterations'],confidence=PARAMS['confidence'],refineIters=10)
        if matrix is None or mask is None or not np.isfinite(matrix).all():break
        good=mask.ravel().astype(bool)
        initial_scale=math.hypot(matrix[0,0],matrix[1,0]);initial_angle=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
        if abs(initial_scale-1)<=PARAMS['translation_scale_deviation']and abs(initial_angle)<=PARAMS['translation_angle']:
            shift=np.median(y[good]-x[good],axis=0);translation_good=np.linalg.norm(y-x-shift,axis=1)<=PARAMS['ransac_px']
            if translation_good.sum()>=max(PARAMS['min_inliers'],PARAMS['translation_support']*good.sum()):
                good=translation_good;matrix=np.float64([[1,0,shift[0]],[0,1,shift[1]]])
        inlier_pairs=[p for p,g in zip(pairs,good)if g];targets={j for i,j in inlier_pairs}
        pairs=[p for p in pairs if p[1]not in targets]
        if min(len({i for i,j in inlier_pairs}),len(targets))<PARAMS['min_inliers']:continue
        xx,yy=x[good],y[good];h,w=a.shape[:2];spans=[np.ptp(z,axis=0)/[w,h]for z in [xx,yy]]
        scale=math.hypot(matrix[0,0],matrix[1,0]);angle=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
        if (any(s[0]<PARAMS['min_span_x']or s[1]<PARAMS['min_span_y']for s in spans)or not PARAMS['scale_min']<=scale<=PARAMS['scale_max']or abs(angle)>PARAMS['angle']or abs(matrix[0,2])>w*PARAMS['translation_fraction']or abs(matrix[1,2])>h*PARAMS['translation_fraction']):continue
        if np.max(np.abs(matrix-np.float64([[1,0,0],[0,1,0]])))<1.:continue
        result.append(dict(matrix=matrix,source_support=hull_mask(xx,(h,w)),target_support=hull_mask(yy,(h,w)),inliers=len(inlier_pairs),source_unique=len({i for i,j in inlier_pairs}),target_unique=len(targets),spans=[s.tolist()for s in spans]))
    return result,counts

def direction(a,b,transforms,reverse=False,debug=False):
    import cv2
    import numpy as np
    h,w=a.shape[:2];a=cv2.GaussianBlur(a,(PARAMS['blur'],PARAMS['blur']),0);b=cv2.GaussianBlur(b,(PARAMS['blur'],PARAMS['blur']),0)
    error=cv2.absdiff(a,b).mean(axis=2);valid=np.ones((h,w),bool);exposed=np.zeros((h,w),bool)
    for t in transforms:
        matrix=cv2.invertAffineTransform(t['matrix'])if reverse else t['matrix'];src=t['target_support']if reverse else t['source_support'];dst=t['source_support']if reverse else t['target_support']
        warped=cv2.warpAffine(a,matrix,(w,h),flags=cv2.INTER_LINEAR)
        domain=cv2.warpAffine(np.ones((h,w),np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
        support=cv2.warpAffine(src.astype(np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST).astype(bool)&domain
        candidate=cv2.absdiff(warped,b).mean(axis=2);error=np.where(support,np.minimum(error,candidate),error)
        valid &= domain
        exposed |= src & ~dst
    eligible=valid & ~exposed;unexplained=(error>PARAMS['residual'])&eligible
    ys,xs=np.nonzero(unexplained)
    result=dict(fraction=float(unexplained.mean()),fit_coverage=float(eligible.mean()),excluded_fit_coverage_fraction=float((~eligible).mean()),bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]if len(xs)else None)
    if debug:return result,error,eligible
    return result

def inspect(a,b):
    """Target-conditioned diagnostic; no automatic state, identity or motion decision."""
    import cv2
    import numpy as np
    if a.shape!=b.shape or a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8 or b.dtype!=np.uint8:raise ValueError('same-size BGR8 required')
    transforms,counts=models(a,b)
    fwd,error,eligible=direction(a,b,transforms,debug=True);back=direction(b,a,transforms,True)
    rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB).copy();unexplained=(error>PARAMS['residual'])&eligible
    rgb[unexplained]=(.4*rgb[unexplained]+.6*np.array([255,0,0])).astype(np.uint8)
    rgb[~eligible]=(.3*rgb[~eligible]+.7*np.array([140,140,140])).astype(np.uint8)
    report=dict(schema='team-eye.appearance-inspection/1',method='bounded multi-warp target-conditioned explanation',parameters=PARAMS,
                coordinate_scope='thumbnail_pixels',shape=list(a.shape[:2]),feature_counts=counts,geometric_model_count=len(transforms),
                geometry_status='unverified_hypotheses'if transforms else'no_nonidentity_model',
                models=[{k:(v.tolist()if k=='matrix'else v)for k,v in t.items()if k not in ['source_support','target_support']}for t in transforms],
                forward=fwd,backward=back,automatic_redraw_decision=None,automatic_action_label=None,used_for_segmentation=False,
                identity_verified=False,anatomical_claims=False,
                limitations=['Alignment/support and fit-coverage exclusion masks are hypotheses, especially at borders or repeated content.',
                             'Residual uses the actual target frame; not a prediction, interpolation or counterfactual.',
                             'Red may be motion, copying, redraw or failed alignment; it is not an outfit or pose-change label.',
                             'Gray marks pixels excluded by the fitted coverage masks, not actual occlusion or source invisibility.',
                             'Explained pixels do not establish object identity; no automatic cut or neural admission decision.'])
    return report,rgb
