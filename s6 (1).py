# study_grid_3d.py
# =============================================================================
# THREE-DIMENSIONAL PARAMETER GRID
# δ*_back  ×  δ*_side  ×  ρ*_int    for 3 CV sizes  ×  5 methods
#
# Grid
# ────
#   δ*_back  : [1.50, 1.00, 0.75, 0.50, 0.375, 0.25]        6 levels
#   δ*_side  : [2.00, 1.20, 0.75, 0.45]                      4 levels
#   ρ*_int   : [0.00, 0.05, 0.10, 0.25]                      4 levels
#   CV sizes : small (2,4,±2)D · medium (2.5,5,±3)D · large (3,6,±4)D
#                                                    = 288 configs × 5 methods
#
# Checkpoint / resume
# ────────────────────
#   Results are saved to  study_grid_3d/results.json  after every grid point.
#   Re-running the script skips already-completed points automatically.
#
# Figure suite  (all saved to study_grid_3d/)
# ────────────────────────────────────────────
#   fig1_overview_{cv}.pdf
#       4 panels (one per ρ* level), each a δ*_back × δ*_side heatmap.
#       Cell colour = BEST drag error across all methods.
#       Cell text   = "X.X% / Method / n=NN".
#
#   fig2_drag_grids_{cv}.pdf
#       5 rows (one per method) × 4 cols (ρ* levels).
#       Each cell: drag error heatmap (δ*_back × δ*_side).
#       Consistent colour scale across methods for fair comparison.
#
#   fig3_rmse_grids_{cv}.pdf
#       Same layout as fig2 but for RMSE [m/s].
#
#   fig4_budget_frontier.pdf
#       n_total vs drag error for every (config, method).
#       Points coloured by method; CV sizes distinguished by marker shape.
#       Threshold lines at 1 %, 2 %, 5 %.
#       Shows the Pareto front clearly.
#
#   fig5_threshold_table.pdf
#       For each method × CV size: minimum n_total to first achieve
#       <1 %, <2 %, <5 % drag error.  Rendered as a colour-coded table.
#
#   fig6_interaction_{cv}.pdf
#       Two interaction plots per CV:
#         (a) mean drag error vs δ*_back, lines coloured by δ*_side
#         (b) mean drag error vs δ*_side, lines coloured by δ*_back
#       Marginalised over ρ* (and per-method panels) — shows which axis
#       matters more.
# =============================================================================

from __future__ import annotations
from pathlib import Path
import hashlib, json, time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import hessian
from scipy.optimize import minimize_scalar, minimize
from scipy.spatial import KDTree, ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.cm as mcm

jax.config.update("jax_enable_x64", True)

HERE       = Path(__file__).resolve().parent
FIELD_PATH = HERE / "output14_5.xlsx"
CYL_PATH   = HERE / "building14_5.xlsx"

D_TRUTH   = 4207.0
RHO       = 1.225
THETA_DEG = -14.5
N_COLL    = 200
N_VAL     = 600
SEED      = "GRID3D"

# ── Grid definition ───────────────────────────────────────────────────────────
GRID_DB = [1.50, 1.00, 0.75, 0.50, 0.375, 0.25]   # δ*_back
GRID_DS = [2.00, 1.20, 0.75, 0.45]                  # δ*_side
GRID_RI = [0.00, 0.05, 0.10, 0.25]                  # ρ*_int

METHOD_KEYS   = ["m1_1d","m2_plain","m3_divfree","m4_cont","m5_bern"]
METHOD_SHORT  = ["M1","M2","M3","M4","M5"]
METHOD_LABELS = ["M1 1D GPR","M2 2D scalar","M3 div-free","M4 continuity","M5 cont+Bern"]
METHOD_COLORS = ["#2e6fa3","#e07b39","#8b5cf6","#4caf7d","#c95f9a"]
METHOD_MKR    = ["o","s","D","^","v"]
CV_MARKERS    = ["o","s","D"]     # small, medium, large

DELTA_SIDE_SAT = 0.65             # used only to compute default n_side (not swept here)


# =============================================================================
# §1  CV + spacing helpers  (identical to v3)
# =============================================================================

class CVGeom:
    def __init__(self,f,b,s): self.front,self.back,self.side=float(f),float(b),float(s)
    @property
    def area(self):     return (self.front+self.back)*2*self.side
    @property
    def back_len(self): return 2*self.side
    @property
    def side_len(self): return self.front+self.back
    def label(self):    return f"({self.front},{self.back},±{self.side})D"

def n_back_from_d(d,cv): return max(3,int(np.ceil(cv.back_len/d))+1)
def n_side_from_d(d,cv): return max(3,int(np.ceil(cv.side_len/d))+1)
def n_int_from_rho(r,cv): return max(0,int(np.ceil(r*cv.area)))
def delta_back(nb,cv):   return cv.back_len/max(1,nb-1)
def _seed(s): return int(hashlib.sha256(s.encode()).hexdigest(),16)%(2**32)


# =============================================================================
# §2  Data loading
# =============================================================================

def _clean(df):
    df.columns=[str(c).strip() for c in df.columns]; return df
def _read(p):
    for e in ("openpyxl","xlrd"):
        try: return _clean(pd.read_excel(p,engine=e))
        except: pass
    raise ValueError(f"Cannot read {p}")

def load_field():
    df=_read(FIELD_PATH); xy=df[["x-coordinate","y-coordinate"]].to_numpy(float)
    uv=df[["x-velocity","y-velocity"]].to_numpy(float)
    for c in ["pressure","static-pressure","Pressure","p","P"]:
        if c in df.columns: return xy,uv,df[c].to_numpy(float)
    raise KeyError("pressure column missing")

def _gdiam(xy):
    try:    h=xy[ConvexHull(xy).vertices]
    except: h=xy
    n=len(h); d=0.; j=1
    for i in range(n):
        while True:
            nj=(j+1)%n
            if np.sum((h[nj]-h[i])**2)>np.sum((h[j]-h[i])**2): j=nj
            else: break
        d=max(d,np.linalg.norm(h[j]-h[i])); j=(j+1)%n
    return float(d)

def load_body():
    df=_read(CYL_PATH); xy=np.unique(df[["x-coordinate","y-coordinate"]].to_numpy(float),axis=0)
    c=xy.mean(0); L=_gdiam(xy)/2.; return xy,c,L

def wall_frame(xy):
    c=xy.mean(0); o=np.argsort(np.arctan2(xy[:,1]-c[1],xy[:,0]-c[0])); xo=xy[o]
    t=np.roll(xo,-1,0)-np.roll(xo,1,0); t/=np.linalg.norm(t,axis=1,keepdims=True)
    n=np.column_stack([t[:,1],-t[:,0]]); sg=np.sign(np.einsum("ij,ij->i",n,xo-c))
    sg[sg==0]=1; n*=sg[:,None]; t*=sg[:,None]
    no=np.empty_like(n); to=np.empty_like(t); no[o]=n; to[o]=t; return no,to

def _R(deg):
    th=np.deg2rad(deg); c,s=np.cos(th),np.sin(th); return np.array([[c,-s],[s,c]])


# =============================================================================
# §3  Drone placement
# =============================================================================

def place_bnd(xy_full,cen,L,th,nf,nb,ns,cv):
    df,db,ds=cv.front,cv.back,cv.side; R=_R(th)
    dd=np.array([1.,0.])@R.T; cd=np.array([0.,1.])@R.T
    pl,nl,fi=[],[],[]
    for v in np.linspace(-ds*.95,ds*.95,nf): pl.append((-df,v)); nl.append((-1.,0.)); fi.append(0)
    for v in np.linspace(-ds*.95,ds*.95,nb): pl.append(( db,v)); nl.append(( 1.,0.)); fi.append(2)
    for u in np.linspace(-df*.95,db*.95,ns): pl.append((u,-ds)); nl.append((0.,-1.)); fi.append(1)
    for u in np.linspace(-df*.95,db*.95,ns): pl.append((u, ds)); nl.append((0., 1.)); fi.append(3)
    pl=np.array(pl); nl=np.array(nl); fi=np.array(fi)
    xt=cen[None,:]+pl[:,0:1]*L*dd[None,:]+pl[:,1:2]*L*cd[None,:]
    nw=nl[:,0:1]*dd[None,:]+nl[:,1:2]*cd[None,:]
    _,ir=KDTree(xy_full).query(xt); _,up=np.unique(ir,return_index=True)
    k=np.sort(up); return ir[k],nw[k],pl[k],fi[k]

def place_int(xy_full,cen,L,n_orb,n_mid,r1=1.8,r2=3.0,tag="I"):
    rng=np.random.default_rng(_seed(tag)); pts=[]
    if n_orb>0:
        ph=rng.uniform(0,2*np.pi); th=ph+np.linspace(0,2*np.pi,n_orb,endpoint=False)
        pts.append(cen+r1*L*np.column_stack([np.cos(th),np.sin(th)]))
    if n_mid>0:
        ph=rng.uniform(0,2*np.pi); th=ph+np.linspace(0,2*np.pi,n_mid,endpoint=False)
        pts.append(cen+r2*L*np.column_stack([np.cos(th),np.sin(th)]))
    if not pts: return np.array([],dtype=int)
    _,idx=KDTree(xy_full).query(np.vstack(pts)); _,u=np.unique(idx,return_index=True)
    return idx[np.sort(u)]

def sample_coll(xy_full,cen,L,n,tag):
    rng=np.random.default_rng(_seed(tag+"C")); pts=[]
    while len(pts)<n:
        r=rng.uniform(1.2*L,6.*L,n*4); th=rng.uniform(0,2*np.pi,n*4)
        pts.extend((cen+np.column_stack([r*np.cos(th),r*np.sin(th)])).tolist())
    return np.array(pts[:n])

def sample_val(xy_full,di,cen,L,n=N_VAL,near=1.2,far=4.,tag="V"):
    rng=np.random.default_rng(_seed(tag)); ds_=set(di.tolist())
    dv=np.linalg.norm(xy_full-cen,axis=1)
    ca=[i for i in range(len(xy_full)) if i not in ds_ and near*L<=dv[i]<=far*L]
    return rng.choice(ca,size=min(n,len(ca)),replace=False)


# =============================================================================
# §4  Kernels  (copy from v3)
# =============================================================================

def safe_chol(K,lbl=""):
    n=K.shape[0]; nu=1e-10
    for _ in range(9):
        try: return np.linalg.cholesky(K+nu*np.eye(n))
        except np.linalg.LinAlgError: nu*=10
    raise np.linalg.LinAlgError(f"[{lbl}] Chol failed")

# 1D non-periodic M52
def _m52_1d(s1,s2,var,ell):
    r=np.abs(s1[:,None]-s2[None,:])/(ell+1e-30); s=np.sqrt(5.)*r
    return var*(1.+s+(s**2)/3.)*np.exp(-s)

def fit_np(s,y,noise=1e-5):
    n=len(s); y=np.asarray(y,float); mu=y.mean(); yc=y-mu
    var=float(np.var(yc)+1e-12); sp=float(s[-1]-s[0]+1e-6)
    def nll(le):
        ell=np.exp(le); K=_m52_1d(s,s,var,ell)+noise*np.eye(n)
        try: Lc=np.linalg.cholesky(K)
        except: return 1e10
        a=np.linalg.solve(Lc.T,np.linalg.solve(Lc,yc))
        return .5*(yc@a+2*np.sum(np.log(np.diag(Lc)))+n*np.log(2*np.pi))
    res=minimize_scalar(nll,bounds=(np.log(.02*sp),np.log(2*sp)),method="bounded")
    ell=float(np.exp(res.x)); K=_m52_1d(s,s,var,ell)+noise*np.eye(n)
    Lc=np.linalg.cholesky(K); a=np.linalg.solve(Lc.T,np.linalg.solve(Lc,yc))
    return dict(s=s,a=a,var=var,ell=ell,mu=mu)

def pred_np(m,sq): return _m52_1d(sq,m["s"],m["var"],m["ell"])@m["a"]+m["mu"]

# 2D scalar M52
def _s52(X1,X2,var,lx,ly):
    dx=X1[:,None,0]-X2[None,:,0]; dy=X1[:,None,1]-X2[None,:,1]
    r=np.sqrt((dx/lx)**2+(dy/ly)**2+1e-12); s=np.sqrt(5.)*r
    return var*(1.+s+(s**2)/3.)*np.exp(-s)

def fit_sc(Xt,yt,noise=1e-5,hp=None,lbl=""):
    if hp is None: hp=dict(var=float(np.var(yt)+1e-12),lx=1.5,ly=1.0)
    K=_s52(Xt,Xt,hp["var"],hp["lx"],hp["ly"])+noise*np.eye(len(Xt))
    Lc=safe_chol(K,lbl); a=np.linalg.solve(Lc.T,np.linalg.solve(Lc,yt))
    return dict(Xt=Xt,a=a,hp=hp)

def pred_sc(m,Xq): return _s52(Xq,m["Xt"],m["hp"]["var"],m["hp"]["lx"],m["hp"]["ly"])@m["a"]

# Div-free (JAX)
def _m52j(x,xp,var,lx,ly):
    dx=(x[0]-xp[0])/lx; dy=(x[1]-xp[1])/ly; r=jnp.sqrt(dx*dx+dy*dy+1e-12); s=jnp.sqrt(5.)*r
    return var*(1.+s+s**2/3.)*jnp.exp(-s)

def _dfe(X,Xp,var,lx,ly):
    x,d=X[:2],X[2:]; xp,dp=Xp[:2],Xp[2:]
    H=-hessian(_m52j,argnums=0)(x,xp,var,lx,ly)
    C=jnp.array([[H[1,1],-H[1,0]],[-H[0,1],H[0,0]]]); return d@C@dp

_dfr=jax.vmap(_dfe,in_axes=(None,0,None,None,None))
dfK =jax.jit(jax.vmap(_dfr,in_axes=(0,None,None,None,None)))

def _to4(xy,uv=None):
    N=xy.shape[0]; pos=jnp.repeat(xy,2,axis=0); dirs=jnp.tile(jnp.eye(2),(N,1))
    X=jnp.hstack([pos,dirs])
    if uv is not None: return X,uv.reshape(-1,order="C")
    return X

# Derivative M52 (continuity/Bernoulli)
def _M52(dx,dy,var,lx,ly):
    r2=(dx/lx)**2+(dy/ly)**2; m=jnp.sqrt(5.*r2+1e-30); E=jnp.exp(-m); P=1.+m+m*m/3.
    K00=var*P*E; f1=-5.*var*(1.+m)*E/3.; K10=f1*dx/lx**2; K01=f1*dy/ly**2; f2=-5.*var*E/3.
    K20=f2*((1.+m)-5.*dx**2/lx**2)/lx**2; K02=f2*((1.+m)-5.*dy**2/ly**2)/ly**2
    K11=25.*var*E*dx*dy/(3.*lx**2*ly**2); return K00,K10,K01,K20,K11,K02

def _df(Kt,al,ar):
    K00,K10,K01,K20,K11,K02=Kt
    return (-1.)**(ar[0]+ar[1])*{(0,0):K00,(1,0):K10,(0,1):K01,
                                   (2,0):K20,(1,1):K11,(0,2):K02}[(al[0]+ar[0],al[1]+ar[1])]

def _ops(): return {"u":[((0,0),"u",1.)],"v":[((0,0),"v",1.)],
                    "p":[((0,0),"p",1.)],"Rc":[((1,0),"u",1.),((0,1),"v",1.)]}

def _Kb(rt,Xr,ct,Xc,huv,hp,ops):
    dx=Xr[:,None,0]-Xc[None,:,0]; dy=Xr[:,None,1]-Xc[None,:,1]
    Kuv=_M52(dx,dy,huv["var"],huv["lx"],huv["ly"]); Kpp=_M52(dx,dy,hp["var"],hp["lx"],hp["ly"])
    out=jnp.zeros_like(dx)
    for(aa,fa,ca)in ops[rt]:
        for(ab,fb,cb)in ops[ct]:
            if fa!=fb: continue
            out=out+(ca*cb)*_df(Kuv if fa in("u","v")else Kpp,aa,ab)
    return out

def _asm(spec,huv,hp,ops):
    cs=[s["count"]for s in spec]; of=np.cumsum([0]+cs); n=of[-1]; K=jnp.zeros((n,n))
    for i,si in enumerate(spec):
        for j,sj in enumerate(spec):
            K=K.at[of[i]:of[i+1],of[j]:of[j+1]].set(_Kb(si["type"],si["X"],sj["type"],sj["X"],huv,hp,ops))
    nd=jnp.concatenate([jnp.full(s["count"],s["noise"])for s in spec]); return K+jnp.diag(nd),of

def _pm(spec,alp,huv,hp,ops,Xq,qt,chunk=4000):
    out=np.zeros(Xq.shape[0])
    for i0 in range(0,Xq.shape[0],chunk):
        i1=min(Xq.shape[0],i0+chunk)
        out[i0:i1]=np.concatenate([np.asarray(_Kb(qt,Xq[i0:i1],s["type"],s["X"],huv,hp,ops))
                                    for s in spec],axis=1)@alp
    return out


# =============================================================================
# §5  Drag + RMSE
# =============================================================================

def cv_dense(cen,L,theta,cv,n=600):
    R=_R(theta); dd=np.array([1.,0.])@R.T; cd=np.array([0.,1.])@R.T
    df,db,ds=cv.front,cv.back,cv.side; lf=2*ds; ls=df+db
    ds_=2*(lf+ls)/n; nf=max(2,int(round(lf/ds_))); ns=max(2,int(round(ls/ds_)))
    pts,nr=[],[]
    for v in np.linspace(-ds,ds,nf,endpoint=False): pts.append(cen+(-df)*L*dd+v*L*cd); nr.append(-dd)
    for v in np.linspace(-ds,ds,nf,endpoint=False): pts.append(cen+ db*L*dd+v*L*cd);   nr.append( dd)
    for u in np.linspace(-df,db,ns,endpoint=False): pts.append(cen+u*L*dd+(-ds)*L*cd); nr.append(-cd)
    for u in np.linspace(-df,db,ns,endpoint=False): pts.append(cen+u*L*dd+  ds*L*cd);  nr.append( cd)
    return np.array(pts),np.array(nr),2*(lf+ls)*L/len(pts)

def drag_from(uf,vf,pf,nq,ds,theta,rho):
    R=_R(theta); dd=np.array([1.,0.])@R.T; uv=np.column_stack([uf,vf])
    nd=nq@dd; ud=uv@dd; un=np.sum(uv*nq,axis=1); return float(np.sum((-pf*nd-rho*ud*un)*ds))

def rmse_vel(u_p,v_p,val_idx,uv_full,U_char):
    eu=u_p-uv_full[val_idx,0]; ev=v_p-uv_full[val_idx,1]
    return float(np.sqrt(np.mean(eu**2+ev**2)))


# =============================================================================
# §6  Method closures  (v3 implementations, condensed)
# =============================================================================

def make_m1(bnd_idx,bnd_nrm,bnd_local,bnd_face,
             xy_full,uv_full,p_full,theta,rho,cv,L,cen,U_inf,V_inf,U_char):
    R=_R(theta); dd=np.array([1.,0.])@R.T; cd=np.array([0.,1.])@R.T
    face_n={0:-dd,1:-cd,2:dd,3:cd}; df,db,ds_=cv.front,cv.back,cv.side
    p_ref=float(np.mean(p_full[bnd_idx])); D=0.
    for fid in [0,1,2,3]:
        m=bnd_face==fid
        if m.sum()<3: continue
        uo=uv_full[bnd_idx[m],0]; vo=uv_full[bnd_idx[m],1]; po=p_full[bnd_idx[m]]-p_ref
        loc=bnd_local[m]; s_raw=loc[:,1]*L if fid in(0,2)else loc[:,0]*L
        o=np.argsort(s_raw); s=s_raw[o]; uo=uo[o]; vo=vo[o]; po=po[o]
        gu=fit_np(s,uo); gv=fit_np(s,vo); gp=fit_np(s,po)
        s_lo=(-ds_*L if fid in(0,2)else -df*L); s_hi=(ds_*L if fid in(0,2)else db*L)
        ni=400; sf=np.linspace(s_lo,s_hi,ni); dsf=(s_hi-s_lo)/(ni-1)
        uf=pred_np(gu,sf); vf=pred_np(gv,sf); pf=pred_np(gp,sf)+p_ref
        nf=np.tile(face_n[fid],(ni,1)); D+=drag_from(uf,vf,pf,nf,dsf,theta,rho)
    xy_nd=(xy_full[bnd_idx]-cen)/L; uv_nd=(uv_full[bnd_idx]-[U_inf,V_inf])/U_char
    p_nd=p_full[bnd_idx]/(rho*U_char**2)
    gu2=fit_sc(xy_nd,uv_nd[:,0],lbl="M1u"); gv2=fit_sc(xy_nd,uv_nd[:,1],lbl="M1v")
    gp2=fit_sc(xy_nd,p_nd,lbl="M1p")
    def pred(xy_q):
        Xq=(xy_q-cen)/L
        return pred_sc(gu2,Xq)*U_char+U_inf, pred_sc(gv2,Xq)*U_char+V_inf, pred_sc(gp2,Xq)*rho*U_char**2
    return D,pred

def make_m2(xy_nd,uv_nd,p_nd,cen,L,U_char,rho,U_inf,V_inf):
    gu=fit_sc(xy_nd,uv_nd[:,0],lbl="M2u"); gv=fit_sc(xy_nd,uv_nd[:,1],lbl="M2v")
    gp=fit_sc(xy_nd,p_nd,lbl="M2p")
    def pred(xy_q):
        Xq=(xy_q-cen)/L
        return pred_sc(gu,Xq)*U_char+U_inf, pred_sc(gv,Xq)*U_char+V_inf, pred_sc(gp,Xq)*rho*U_char**2
    return pred

def make_m3(xy_nd,uv_nd,p_nd,cen,L,U_char,rho,U_inf,V_inf,theta_fixed=None):
    JITTER=1e-8; LX_MIN=np.log(.25)
    X_v,y_v=_to4(jnp.array(xy_nd),jnp.array(uv_nd)); mv=jnp.ones(len(y_v))
    bds=((-6,6),(LX_MIN,2.5),(LX_MIN,2.5),(-12,2))
    starts=[np.array([0.,1.,1.,-4.]),np.array([1.,.5,.5,-3.]),
            np.array([-.5,-.5,-.5,-5.]),np.array([.5,2.,.5,-4.])]
    if theta_fixed is None:
        def _nll(th):
            th=jnp.array(th); vs,lx,ly,noise=jnp.exp(th)
            K=dfK(X_v,X_v,vs,lx,ly)+jnp.diag(noise*mv+JITTER*(1-mv))
            Lc=jnp.linalg.cholesky(K); a=jax.scipy.linalg.cho_solve((Lc,True),y_v)
            return float(.5*(y_v@a+2*jnp.sum(jnp.log(jnp.diag(Lc)))+len(y_v)*jnp.log(2*jnp.pi)))
        best=None
        for s0 in starts:
            try:
                r=minimize(_nll,x0=s0,method="L-BFGS-B",bounds=bds)
                if np.isfinite(r.fun)and(best is None or r.fun<best.fun): best=r
            except: pass
        theta=jnp.array(best.x)
    else: theta=jnp.array(theta_fixed)
    vs,lx,ly,noise=jnp.exp(theta); K_f=dfK(X_v,X_v,vs,lx,ly)+jnp.diag(noise*mv+JITTER*(1-mv))
    Lc_f=jnp.linalg.cholesky(K_f); alp=np.array(jax.scipy.linalg.cho_solve((Lc_f,True),y_v))
    gp_p=fit_sc(xy_nd,p_nd,lbl="M3p")
    def pred(xy_q):
        Xq=(xy_q-cen)/L; Xt=_to4(jnp.array(Xq)); pr=[]
        for i0 in range(0,len(Xt),2000): pr.append(np.array(dfK(Xt[i0:i0+2000],X_v,vs,lx,ly)@alp))
        uv_q=np.concatenate(pr).reshape(-1,2,order="C"); pp=pred_sc(gp_p,(xy_q-cen)/L)*rho*U_char**2
        return uv_q[:,0]*U_char+U_inf, uv_q[:,1]*U_char+V_inf, pp
    return pred,np.array(theta)

def make_m45(xy_s,uv_s,p_s,xy_coll,U_inf,V_inf,U_char,L,cen,rho,cv,method):
    hp_uv=dict(var=.5,lx=max(1.,cv.back),ly=max(.8,cv.side*.5))
    hp_p =dict(var=.1,lx=max(.8,cv.back*.8),ly=max(.6,cv.side*.4))
    tf=float(np.arctan2(V_inf,U_inf)); c_,s_=np.cos(tf),np.sin(tf)
    RAB=np.array([[c_,s_],[-s_,c_]]); RBA=np.array([[c_,-s_],[s_,c_]])
    def ndxy(X): return ((X-cen)/L)@RAB.T
    def nduv(V): return ((V-np.array([U_inf,V_inf]))/U_char)@RAB.T
    def ndp(P):  return P/(rho*U_char**2)
    ops=_ops(); Xs=jnp.array(ndxy(xy_s)); uvn=nduv(uv_s); pn=ndp(p_s)
    spec=[dict(name="u",type="u",X=Xs,count=len(xy_s),noise=1e-6),
          dict(name="v",type="v",X=Xs,count=len(xy_s),noise=1e-6),
          dict(name="p",type="p",X=Xs,count=len(xy_s),noise=1e-6)]
    yp=[jnp.array(uvn[:,0]),jnp.array(uvn[:,1]),jnp.array(pn)]
    if method=="cont_anchor_bern":
        anb=np.array([[-5.,0.]]); xya=cen+L*(anb@RBA.T)
        spec.append(dict(name="pa",type="p",X=jnp.array(ndxy(xya)),count=1,noise=1e-10)); yp.append(jnp.zeros(1))
    Xc=jnp.array(ndxy(xy_coll))
    spec.append(dict(name="Rc",type="Rc",X=Xc,count=len(xy_coll),noise=1e-4)); yp.append(jnp.zeros(len(xy_coll)))
    if method=="cont_anchor_bern":
        bpf=max(1.5,cv.front*2*.8); bws=max(1.5,cv.back*2*.25)
        yt1=jnp.concatenate(yp); K1,_=_asm(spec,hp_uv,hp_p,ops)
        K1=np.asarray(K1+1e-6*jnp.eye(K1.shape[0])); L1=safe_chol(K1,"M5p1")
        a1=np.linalg.solve(L1.T,np.linalg.solve(L1,np.asarray(yt1)))
        loc=(xy_coll-cen)@RAB.T/L; rl=np.hypot(loc[:,0],loc[:,1])
        in_wk=(loc[:,0]>bws)&(np.arctan2(np.abs(loc[:,1]),loc[:,0])*180/np.pi<25.)
        msk=(rl>bpf)&(~in_wk)
        if msk.any():
            Xb=jnp.array(ndxy(xy_coll[msk]))
            ub=_pm(spec,a1,hp_uv,hp_p,ops,Xb,"u"); vb=_pm(spec,a1,hp_uv,hp_p,ops,Xb,"v")
            sp2=np.sum(np.column_stack([ub+1.,vb])**2,axis=1)
            spec.append(dict(name="bn",type="p",X=Xb,count=len(Xb),noise=.02)); yp.append(jnp.array(.5*(1.-sp2)))
    ytr=jnp.concatenate(yp); Ktr,_=_asm(spec,hp_uv,hp_p,ops)
    Ktr=np.asarray(Ktr+1e-6*jnp.eye(Ktr.shape[0])); Lc=safe_chol(Ktr,"M4"if method=="cont"else"M5")
    alp=np.linalg.solve(Lc.T,np.linalg.solve(Lc,np.asarray(ytr)))
    def pred(xy_q):
        Xq=jnp.array(ndxy(xy_q)); uB=_pm(spec,alp,hp_uv,hp_p,ops,Xq,"u")
        vB=_pm(spec,alp,hp_uv,hp_p,ops,Xq,"v"); pp=_pm(spec,alp,hp_uv,hp_p,ops,Xq,"p")
        uvA=np.column_stack([uB,vB])@RBA.T
        return uvA[:,0]*U_char+U_inf, uvA[:,1]*U_char+V_inf, pp*rho*U_char**2
    return pred


# =============================================================================
# §7  Single grid-point evaluation
# =============================================================================

def eval_point(xy_full,uv_full,p_full,cen,L,U_inf,V_inf,U_char,
               bnd_idx,bnd_nrm,bnd_local,bnd_face,interior_idx,
               rho,xy_coll,val_idx,cv,n_back,m3_cache):
    """Returns dict: {method_key: {drag_err, rmse_vel_ms}}"""
    all_idx=(np.concatenate([bnd_idx,interior_idx]) if len(interior_idx) else bnd_idx.copy())
    xy_q,nq,ds_cv=cv_dense(cen,L,THETA_DEG,cv)
    xy_val=xy_full[val_idx]
    xy_nd=(xy_full[all_idx]-cen)/L; uv_nd=(uv_full[all_idx]-[U_inf,V_inf])/U_char
    p_nd=p_full[all_idx]/(rho*U_char**2)
    res={}

    def _store(mk,D,u_v,v_v):
        de=100.*abs(D-D_TRUTH)/abs(D_TRUTH); rm=rmse_vel(u_v,v_v,val_idx,uv_full,U_char)
        res[mk]=dict(drag_err=float(de),rmse_ms=float(rm))

    # M1
    try:
        D1,pf1=make_m1(bnd_idx,bnd_nrm,bnd_local,bnd_face,
                        xy_full,uv_full,p_full,THETA_DEG,rho,cv,L,cen,U_inf,V_inf,U_char)
        u1v,v1v,_=pf1(xy_val); _store("m1_1d",D1,u1v,v1v)
    except Exception as e: print(f"    M1 FAIL {e}"); res["m1_1d"]=dict(drag_err=np.nan,rmse_ms=np.nan)

    # M2
    try:
        pf2=make_m2(xy_nd,uv_nd,p_nd,cen,L,U_char,rho,U_inf,V_inf)
        u2q,v2q,p2q=pf2(xy_q); u2v,v2v,_=pf2(xy_val)
        _store("m2_plain",drag_from(u2q,v2q,p2q,nq,ds_cv,THETA_DEG,rho),u2v,v2v)
    except Exception as e: print(f"    M2 FAIL {e}"); res["m2_plain"]=dict(drag_err=np.nan,rmse_ms=np.nan)

    # M3 (HP cached)
    try:
        th_in=m3_cache.get("theta")
        pf3,th_out=make_m3(xy_nd,uv_nd,p_nd,cen,L,U_char,rho,U_inf,V_inf,theta_fixed=th_in)
        if "theta" not in m3_cache: m3_cache["theta"]=th_out
        u3q,v3q,p3q=pf3(xy_q); u3v,v3v,_=pf3(xy_val)
        _store("m3_divfree",drag_from(u3q,v3q,p3q,nq,ds_cv,THETA_DEG,rho),u3v,v3v)
    except Exception as e: print(f"    M3 FAIL {e}"); res["m3_divfree"]=dict(drag_err=np.nan,rmse_ms=np.nan)

    # M4
    try:
        pf4=make_m45(xy_full[all_idx],uv_full[all_idx],p_full[all_idx],
                      xy_coll,U_inf,V_inf,U_char,L,cen,rho,cv,"cont")
        u4q,v4q,p4q=pf4(xy_q); u4v,v4v,_=pf4(xy_val)
        _store("m4_cont",drag_from(u4q,v4q,p4q,nq,ds_cv,THETA_DEG,rho),u4v,v4v)
    except Exception as e: print(f"    M4 FAIL {e}"); res["m4_cont"]=dict(drag_err=np.nan,rmse_ms=np.nan)

    # M5
    try:
        pf5=make_m45(xy_full[all_idx],uv_full[all_idx],p_full[all_idx],
                      xy_coll,U_inf,V_inf,U_char,L,cen,rho,cv,"cont_anchor_bern")
        u5q,v5q,p5q=pf5(xy_q); u5v,v5v,_=pf5(xy_val)
        _store("m5_bern",drag_from(u5q,v5q,p5q,nq,ds_cv,THETA_DEG,rho),u5v,v5v)
    except Exception as e: print(f"    M5 FAIL {e}"); res["m5_bern"]=dict(drag_err=np.nan,rmse_ms=np.nan)

    return res


# =============================================================================
# §8  CHECKPOINT  (save/load JSON after every grid point)
# =============================================================================

def _pt_key(cv_label,i_db,i_ds,i_ri): return f"{cv_label}|{i_db}|{i_ds}|{i_ri}"

def load_checkpoint(path):
    if path.exists():
        with open(path) as f: data=json.load(f)
        print(f"  Resuming: {len(data)} points already done.")
        return data
    return {}

def save_checkpoint(data,path):
    with open(path,"w") as f: json.dump(data,f,indent=2)


# =============================================================================
# §9  MAIN GRID RUNNER
# =============================================================================

def run_grid(xy_full,uv_full,p_full,cen,L,U_inf,V_inf,U_char,xy_coll,
              cv,cv_name,results,ckpt_path,n_front=5):
    """
    Run 3D grid for one CV size.  Writes results in-place to `results` dict.
    Skips already-completed points.  Saves checkpoint after every point.
    """
    total=len(GRID_DB)*len(GRID_DS)*len(GRID_RI); done=0; t0=time.time()
    m3_cache={}   # HP fixed across the whole grid for this CV

    for i_db,db in enumerate(GRID_DB):
        nb=n_back_from_d(db,cv)
        for i_ds,ds_ in enumerate(GRID_DS):
            ns=n_side_from_d(ds_,cv)
            # Place boundary (fixed for all ρ* in this row)
            bi,bn,bl,bf=place_bnd(xy_full,cen,L,THETA_DEG,n_front,nb,ns,cv)

            for i_ri,ri in enumerate(GRID_RI):
                key=_pt_key(cv.label(),i_db,i_ds,i_ri)
                if key in results:
                    done+=1; continue  # already done

                ni=n_int_from_rho(ri,cv)
                n_orb=max(0,int(round(ni*2/3))); n_mid=max(0,ni-n_orb)
                int_idx=(place_int(xy_full,cen,L,n_orb,n_mid,
                                    tag=f"G{cv_name}{i_db}{i_ds}{i_ri}")
                          if ni>0 else np.array([],dtype=int))
                all_d=np.concatenate([bi,int_idx]) if len(int_idx) else bi.copy()
                vi=sample_val(xy_full,all_d,cen,L,tag=f"Gv{cv_name}{i_db}{i_ds}{i_ri}")

                n_total=len(all_d); actual_db=delta_back(nb,cv)
                actual_ds=cv.side_len/max(1,ns-1)
                actual_ri=ni/cv.area if cv.area>0 else 0.

                print(f"  [{cv_name}] δb={actual_db:.3f} δs={actual_ds:.3f} "
                      f"ρ={actual_ri:.3f}  n={n_total}  ...",flush=True)

                pt_res=eval_point(xy_full,uv_full,p_full,cen,L,U_inf,V_inf,U_char,
                                   bi,bn,bl,bf,int_idx,RHO,xy_coll,vi,cv,nb,m3_cache)

                results[key]=dict(
                    cv=cv.label(), cv_name=cv_name,
                    i_db=i_db, i_ds=i_ds, i_ri=i_ri,
                    db_target=db, ds_target=ds_, ri_target=ri,
                    db_actual=float(actual_db), ds_actual=float(actual_ds),
                    ri_actual=float(actual_ri),
                    n_back=int(nb), n_side=int(ns), n_int=int(ni),
                    n_total=int(n_total),
                    methods=pt_res
                )
                done+=1; save_checkpoint(results,ckpt_path)
                elapsed=time.time()-t0
                rate=done/elapsed if elapsed>0 else 1
                remain=(total*len(list({k.split("|")[0] for k in results.keys()}))
                        - done) / max(rate,1)
                drag_str=" ".join(f"{k[:2]}={v['drag_err']:.1f}%"
                                   for k,v in pt_res.items())
                print(f"    {drag_str}  [ETA {remain/60:.0f} min]")


# =============================================================================
# §10  FIGURE GENERATORS
# =============================================================================

def _heatmap(ax, mat, row_labels, col_labels, ann_mat,
              vmin, vmax, cmap, xlabel, ylabel, title, fmt=".1f"):
    """Generic annotated heatmap on a given axes."""
    im=ax.imshow(mat,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto",
                  norm=mcolors.PowerNorm(gamma=.6,vmin=vmin,vmax=vmax))
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels,fontsize=7)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels,fontsize=7)
    ax.set_xlabel(xlabel,fontsize=8); ax.set_ylabel(ylabel,fontsize=8)
    ax.set_title(title,fontsize=8,pad=3)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v=mat[r,c]
            if not np.isnan(v):
                txt=ann_mat[r,c] if ann_mat is not None else f"{v:{fmt}}"
                col_txt="white" if v>vmax*.6 else "black"
                ax.text(c,r,txt,ha="center",va="center",fontsize=6,color=col_txt)
    return im


def _get_slice(results,cv_label,i_db,i_ds,i_ri,method,metric):
    key=_pt_key(cv_label,i_db,i_ds,i_ri)
    if key not in results: return np.nan
    v=results[key]["methods"].get(method,{}).get(metric,np.nan)
    return float(v) if v is not None else np.nan


def fig1_overview(results,cv,out):
    """
    Best-method heatmap for each ρ* level.
    δ*_back on Y, δ*_side on X.  Colour = best drag error across methods.
    Cell text = "drag%\nMethodName\nn=NN"
    """
    n_ri=len(GRID_RI); n_db=len(GRID_DB); n_ds=len(GRID_DS)
    fig,axes=plt.subplots(1,n_ri,figsize=(5.5*n_ri,6),squeeze=False)
    yl=[f"{db:.2f}" for db in GRID_DB]; xl=[f"{ds:.2f}" for ds in GRID_DS]

    for j,ri in enumerate(GRID_RI):
        ax=axes[0,j]; best_drag=np.full((n_db,n_ds),np.nan); ann=np.empty((n_db,n_ds),dtype=object)
        for i_db in range(n_db):
            for i_ds in range(n_ds):
                key=_pt_key(cv.label(),i_db,i_ds,j)
                if key not in results: continue
                pt=results[key]
                drags={mk:pt["methods"][mk]["drag_err"] for mk in METHOD_KEYS
                        if not np.isnan(pt["methods"].get(mk,{}).get("drag_err",np.nan))}
                if not drags: continue
                best_mk=min(drags,key=drags.get); best_v=drags[best_mk]
                best_drag[i_db,i_ds]=best_v
                mshort=METHOD_SHORT[METHOD_KEYS.index(best_mk)]
                ann[i_db,i_ds]=f"{best_v:.1f}%\n{mshort}\nn={pt['n_total']}"
        im=_heatmap(ax,best_drag,yl,xl,ann,0,20,"RdYlGn_r",
                     r"$\delta^*_\mathrm{side}$",r"$\delta^*_\mathrm{back}$",
                     f"ρ*={ri:.2f} — best drag error (%)")
        ax.invert_yaxis()
        for thresh,col in [(1.,"#00aa00"),(2.,"#aaaa00"),(5.,"#aa0000")]:
            ax.contour(best_drag,levels=[thresh],colors=[col],linewidths=1.5,linestyles="--")
    plt.colorbar(im,ax=axes[0,:].tolist(),shrink=.7,label="Best drag error (%)",location="right")
    fig.suptitle(f"Overview — CV {cv.label()}\n"
                  "Each cell: best drag error across all methods  |  dashed = 1%, 2%, 5% thresholds",
                  fontsize=11)
    fig.tight_layout(); fig.savefig(out/f"fig1_overview_{cv.label().replace(',','_')}.pdf",bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1 saved for {cv.label()}")


def fig2_drag_grids(results,cv,out):
    """
    Per-method drag error heatmaps.
    Rows = methods, cols = ρ* levels.  Each cell = heatmap(δ*_back × δ*_side).
    """
    n_m=len(METHOD_KEYS); n_ri=len(GRID_RI)
    fig,axes=plt.subplots(n_m,n_ri,figsize=(4.5*n_ri,4*n_m),squeeze=False)
    yl=[f"{db:.2f}" for db in GRID_DB]; xl=[f"{ds:.2f}" for ds in GRID_DS]

    # Compute shared vmax (cap at 30%)
    vmax=0.
    for mk in METHOD_KEYS:
        for i_ri in range(n_ri):
            for i_db in range(len(GRID_DB)):
                for i_ds in range(len(GRID_DS)):
                    v=_get_slice(results,cv.label(),i_db,i_ds,i_ri,mk,"drag_err")
                    if not np.isnan(v) and v<50: vmax=max(vmax,v)
    vmax=min(vmax,30.)

    for mi,mk in enumerate(METHOD_KEYS):
        for j,ri in enumerate(GRID_RI):
            ax=axes[mi,j]; mat=np.full((len(GRID_DB),len(GRID_DS)),np.nan)
            ann=np.empty_like(mat,dtype=object)
            for i_db in range(len(GRID_DB)):
                for i_ds in range(len(GRID_DS)):
                    v=_get_slice(results,cv.label(),i_db,i_ds,j,mk,"drag_err")
                    mat[i_db,i_ds]=v
                    key=_pt_key(cv.label(),i_db,i_ds,j)
                    nt=results[key]["n_total"] if key in results else "?"
                    ann[i_db,i_ds]=(f"{v:.1f}%\n(n={nt})" if not np.isnan(v) else "—")
            title=(f"{METHOD_LABELS[mi]}  ρ*={ri:.2f}"
                   if j==0 else f"ρ*={ri:.2f}")
            ylabel=r"$\delta^*_\mathrm{back}$" if j==0 else ""
            xlabel=r"$\delta^*_\mathrm{side}$" if mi==n_m-1 else ""
            im=_heatmap(ax,mat,yl,xl,ann,0,vmax,"RdYlGn_r",xlabel,ylabel,title)
            ax.invert_yaxis()
            if j==0: ax.set_ylabel(f"{METHOD_SHORT[mi]}\n"+r"$\delta^*_\mathrm{back}$",fontsize=8)
    plt.colorbar(im,ax=axes[:,n_ri-1].tolist(),shrink=.7,label="Drag error (%)",location="right")
    fig.suptitle(f"Drag error grids — CV {cv.label()}\n"
                  "Rows = methods | Cols = interior density ρ*",fontsize=11)
    fig.tight_layout(); fig.savefig(out/f"fig2_drag_{cv.label().replace(',','_')}.pdf",bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2 saved for {cv.label()}")


def fig3_rmse_grids(results,cv,out):
    """Per-method RMSE [m/s] heatmaps.  Same layout as fig2."""
    n_m=len(METHOD_KEYS); n_ri=len(GRID_RI)
    fig,axes=plt.subplots(n_m,n_ri,figsize=(4.5*n_ri,4*n_m),squeeze=False)
    yl=[f"{db:.2f}" for db in GRID_DB]; xl=[f"{ds:.2f}" for ds in GRID_DS]
    vmax=0.
    for mk in METHOD_KEYS:
        for i_ri in range(n_ri):
            for i_db in range(len(GRID_DB)):
                for i_ds in range(len(GRID_DS)):
                    v=_get_slice(results,cv.label(),i_db,i_ds,i_ri,mk,"rmse_ms")
                    if not np.isnan(v): vmax=max(vmax,v)
    vmax=min(vmax,4.0)
    for mi,mk in enumerate(METHOD_KEYS):
        for j,ri in enumerate(GRID_RI):
            ax=axes[mi,j]; mat=np.full((len(GRID_DB),len(GRID_DS)),np.nan)
            ann=np.empty_like(mat,dtype=object)
            for i_db in range(len(GRID_DB)):
                for i_ds in range(len(GRID_DS)):
                    v=_get_slice(results,cv.label(),i_db,i_ds,j,mk,"rmse_ms")
                    mat[i_db,i_ds]=v
                    ann[i_db,i_ds]=(f"{v:.2f}" if not np.isnan(v) else "—")
            title=(f"{METHOD_LABELS[mi]}  ρ*={ri:.2f}" if j==0 else f"ρ*={ri:.2f}")
            ylabel=r"$\delta^*_\mathrm{back}$" if j==0 else ""
            xlabel=r"$\delta^*_\mathrm{side}$" if mi==n_m-1 else ""
            im=_heatmap(ax,mat,yl,xl,ann,0,vmax,"plasma_r",xlabel,ylabel,title)
            ax.invert_yaxis()
            if j==0: ax.set_ylabel(f"{METHOD_SHORT[mi]}\n"+r"$\delta^*_\mathrm{back}$",fontsize=8)
    plt.colorbar(im,ax=axes[:,n_ri-1].tolist(),shrink=.7,label="RMSE [m/s]",location="right")
    fig.suptitle(f"RMSE grids [m/s] — CV {cv.label()}\n"
                  "Rows = methods | Cols = interior density ρ*",fontsize=11)
    fig.tight_layout(); fig.savefig(out/f"fig3_rmse_{cv.label().replace(',','_')}.pdf",bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3 saved for {cv.label()}")


def fig4_budget_frontier(results,cvs,out):
    """
    n_total vs drag error for every (config, method, CV).
    Log-y scale.  Pareto-optimal frontier highlighted per method.
    """
    fig,ax=plt.subplots(figsize=(13,7))
    cm_cv=plt.cm.Set2(np.linspace(0,.8,len(cvs)))

    for ci,(cv_name,cv) in enumerate(cvs.items()):
        for mi,(mk,col,mkr) in enumerate(zip(METHOD_KEYS,METHOD_COLORS,METHOD_MKR)):
            pts=[]
            for key,rec in results.items():
                if rec["cv"]!=cv.label(): continue
                v=rec["methods"].get(mk,{})
                de=v.get("drag_err",np.nan)
                if np.isnan(de) or de>50: continue
                pts.append((rec["n_total"],de))
            if not pts: continue
            pts.sort(); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            label=(f"{METHOD_SHORT[mi]} {cv_name}" if ci==0 else None)
            ax.scatter(xs,ys,color=col,marker=CV_MARKERS[ci],s=30,alpha=.55,
                        edgecolors=cm_cv[ci],linewidths=.6)
            # Pareto front
            pareto=[]; best=np.inf
            for x,y in sorted(pts):
                if y<best: pareto.append((x,y)); best=y
            if len(pareto)>1:
                px,py=zip(*pareto)
                ax.plot(px,py,color=col,ls="-",lw=1.4,alpha=.8,
                         label=f"{METHOD_SHORT[mi]} {cv_name}")

    for thresh,col,ls in [(1.,"#00aa00","--"),(2.,"#aaaa00","-."),(5.,"#cc3300",":")]:
        ax.axhline(thresh,color=col,lw=1.5,ls=ls,label=f"{thresh}% target")

    ax.set_yscale("log"); ax.set_xlabel("Total drones  $n_\\mathrm{total}$",fontsize=11)
    ax.set_ylabel("Drag error (%)  [log scale]",fontsize=11)
    ax.set_title("Budget frontier — Pareto-optimal (n_total, drag error) per method × CV size\n"
                  "Lines = Pareto front.  Points = all grid configs.",fontsize=11)
    ax.grid(True,which="both",ls=":",alpha=.4); ax.legend(fontsize=7,ncol=4,loc="upper right")
    ax.set_ylim(.1,55)
    fig.tight_layout(); fig.savefig(out/"fig4_budget_frontier.pdf",bbox_inches="tight")
    plt.close(fig); print("  fig4 saved")


def fig5_threshold_table(results,cvs,out):
    """
    Colour-coded table: minimum n_total to first achieve <1%, <2%, <5% drag.
    Rows = method × CV size combinations.  Columns = drag thresholds.
    """
    thresholds=[1.,2.,5.]; t_labels=["<1%","<2%","<5%"]
    row_labels=[]; data=[]
    for cv_name,cv in cvs.items():
        for mk,mlab in zip(METHOD_KEYS,METHOD_SHORT):
            row_labels.append(f"{cv_name}\n{mlab}")
            row=[]
            for th in thresholds:
                cands=[rec["n_total"] for key,rec in results.items()
                        if rec["cv"]==cv.label()
                        and not np.isnan(rec["methods"].get(mk,{}).get("drag_err",np.nan))
                        and rec["methods"][mk]["drag_err"]<th]
                row.append(min(cands) if cands else np.nan)
            data.append(row)
    data=np.array(data,dtype=float)

    fig,ax=plt.subplots(figsize=(8,max(5,len(row_labels)*.4+2)))
    ax.axis("off")
    vmin=float(np.nanmin(data)); vmax=float(np.nanmax(data))
    norm=mcolors.Normalize(vmin=vmin,vmax=vmax)
    cmap=mcm.get_cmap("RdYlGn_r")
    tbl=ax.table(
        cellText=[[("—" if np.isnan(v) else str(int(v))) for v in row] for row in data],
        rowLabels=row_labels, colLabels=t_labels,
        cellLoc="center", rowLoc="center", loc="center",
        bbox=[.15,.02,.82,.96])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    # Colour cells
    for ri,row in enumerate(data):
        for ci,v in enumerate(row):
            if not np.isnan(v):
                rgb=cmap(norm(v)); tbl[(ri+1,ci)].set_facecolor(rgb)
                tbl[(ri+1,ci)].set_text_props(
                    color="white" if norm(v)>.6 else "black")
    # Bold header
    for ci in range(len(thresholds)): tbl[(0,ci)].set_text_props(fontweight="bold")
    # Separator lines between CV groups
    n_m=len(METHOD_KEYS)
    for gi in range(1,len(cvs)):
        for ci in range(len(thresholds)):
            tbl[(gi*n_m+1,ci)].set_edgecolor("#444444"); tbl[(gi*n_m+1,ci)].set_linewidth(2.)
    ax.set_title("Minimum n_total to achieve drag error threshold\n"
                  "(colour: green=few drones, red=many)",fontsize=11,pad=10)
    fig.tight_layout(); fig.savefig(out/"fig5_threshold_table.pdf",bbox_inches="tight")
    plt.close(fig); print("  fig5 saved")


def fig6_interactions(results,cv,out):
    """
    Interaction plots: mean drag vs δ*_back (lines = δ*_side levels)
    and mean drag vs δ*_side (lines = δ*_back levels).
    One row per method.
    """
    n_m=len(METHOD_KEYS); fig,axes=plt.subplots(n_m,2,figsize=(13,3.5*n_m),squeeze=False)
    cmap_b=plt.cm.Blues_r(np.linspace(.2,.8,len(GRID_DB)))
    cmap_s=plt.cm.Reds_r(np.linspace(.2,.8,len(GRID_DS)))

    for mi,mk in enumerate(METHOD_KEYS):
        # Panel A: mean drag vs δ*_back, lines = δ*_side
        ax_a=axes[mi,0]
        for i_ds,ds_ in enumerate(GRID_DS):
            means=[]
            for i_db,db in enumerate(GRID_DB):
                vals=[_get_slice(results,cv.label(),i_db,i_ds,i_ri,mk,"drag_err")
                       for i_ri in range(len(GRID_RI))]
                vals=[v for v in vals if not np.isnan(v)]
                means.append(np.mean(vals) if vals else np.nan)
            ax_a.plot(GRID_DB,means,color=cmap_s[i_ds],lw=2.,marker="o",ms=4,
                       label=f"δs={ds_:.2f}")
        ax_a.set_xlabel(r"$\delta^*_\mathrm{back}$",fontsize=9)
        ax_a.set_ylabel("Mean drag error (%)",fontsize=8)
        ax_a.set_title(f"{METHOD_LABELS[mi]} — effect of back-face spacing\n"
                        "(each line = one δ*_side level, averaged over ρ*)",fontsize=8)
        ax_a.invert_xaxis(); ax_a.grid(True,ls=":",alpha=.4)
        ax_a.axhline(1.,color="green",lw=1.,ls="--",alpha=.5)
        ax_a.legend(fontsize=6,ncol=2)

        # Panel B: mean drag vs δ*_side, lines = δ*_back
        ax_b=axes[mi,1]
        for i_db,db in enumerate(GRID_DB):
            means=[]
            for i_ds in range(len(GRID_DS)):
                vals=[_get_slice(results,cv.label(),i_db,i_ds,i_ri,mk,"drag_err")
                       for i_ri in range(len(GRID_RI))]
                vals=[v for v in vals if not np.isnan(v)]
                means.append(np.mean(vals) if vals else np.nan)
            ax_b.plot(GRID_DS,means,color=cmap_b[i_db],lw=2.,marker="s",ms=4,
                       label=f"δb={db:.2f}")
        ax_b.set_xlabel(r"$\delta^*_\mathrm{side}$",fontsize=9)
        ax_b.set_ylabel("Mean drag error (%)",fontsize=8)
        ax_b.set_title(f"{METHOD_LABELS[mi]} — effect of side-face spacing\n"
                        "(each line = one δ*_back level, averaged over ρ*)",fontsize=8)
        ax_b.invert_xaxis(); ax_b.grid(True,ls=":",alpha=.4)
        ax_b.axhline(1.,color="green",lw=1.,ls="--",alpha=.5)
        ax_b.legend(fontsize=6,ncol=2)

    fig.suptitle(f"Interaction plots — CV {cv.label()}\n"
                  "Left: δ*_back dominates?  Right: δ*_side dominates?",fontsize=11)
    fig.tight_layout(); fig.savefig(out/f"fig6_interaction_{cv.label().replace(',','_')}.pdf",bbox_inches="tight")
    plt.close(fig)
    print(f"  fig6 saved for {cv.label()}")


# =============================================================================
# §11  MAIN
# =============================================================================

def run():
    print("="*72); print(" 3D GRID STUDY"); print("="*72)
    print(f" Grid: {len(GRID_DB)} δ*_back × {len(GRID_DS)} δ*_side × {len(GRID_RI)} ρ*_int")
    print(f"       = {len(GRID_DB)*len(GRID_DS)*len(GRID_RI)} configs × 3 CVs × 5 methods")

    xy_full,uv_full,p_full=load_field()
    xy_body,cen,L=load_body()
    up=xy_full[:,0]<(cen[0]-3*L)
    U_inf=float(uv_full[up,0].mean()); V_inf=float(uv_full[up,1].mean())
    U_char=float(np.hypot(U_inf,V_inf))
    print(f" L={L:.3f} m  U_char={U_char:.3f} m/s")

    xy_coll=sample_coll(xy_full,cen,L,N_COLL,SEED)

    cvs=dict(small =CVGeom(2.,4.,2.),
             medium=CVGeom(2.5,5.,3.),
             large =CVGeom(3.,6.,4.))

    out=HERE/"study_grid_3d"; out.mkdir(exist_ok=True)
    ckpt=out/"results.json"
    results=load_checkpoint(ckpt)

    # ── Compute ──────────────────────────────────────────────────────────────
    for cv_name,cv in cvs.items():
        run_grid(xy_full,uv_full,p_full,cen,L,U_inf,V_inf,U_char,
                  xy_coll,cv,cv_name,results,ckpt)

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    for cv_name,cv in cvs.items():
        fig1_overview(results,cv,out)
        fig2_drag_grids(results,cv,out)
        fig3_rmse_grids(results,cv,out)
        fig6_interactions(results,cv,out)

    fig4_budget_frontier(results,cvs,out)
    fig5_threshold_table(results,cvs,out)

    pdfs=list(out.glob("*.pdf"))
    print(f"\n Done.  {len(pdfs)} figures saved to {out}/")
    print(" Figure guide:")
    print("   fig1_overview_*.pdf      — best method at each (δb,δs,ρ) grid point")
    print("   fig2_drag_*.pdf          — drag error heatmaps per method")
    print("   fig3_rmse_*.pdf          — RMSE [m/s] heatmaps per method")
    print("   fig4_budget_frontier.pdf — n_total vs drag: Pareto fronts")
    print("   fig5_threshold_table.pdf — min drones to hit <1%/<2%/<5% drag")
    print("   fig6_interaction_*.pdf   — which spacing axis matters most?")
    return results


if __name__=="__main__":
    run()