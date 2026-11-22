#!/usr/bin/env python3
"""
report.py - Driftlock mission report: one self-contained HTML page per run.

  python tools/report.py <degraded imu.csv> [--gt groundtruth.csv] [-o report.html] [--title "..."]

Runs the estimator, then writes an interactive page (no internet needed):
  * 3D flight path you can drag to rotate, with play/pause and a time scrubber
  * the blind 5 s highlighted, and (with --gt) the hidden ground truth + scores
  * a forensics timeline: dead sensor, gaps, duplicates, frozen rows, spikes
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import main  # noqa: E402
import evaluate  # noqa: E402


def downsample(n, k=1500):
    return np.unique(np.linspace(0, n - 1, min(n, k)).astype(int))


def build(imu, gt=None, title=None):
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "est.txt")
    t0 = time.perf_counter()
    ts, raw = main.read_imu(imu)
    gyro_norm = np.linalg.norm(raw[:, 0:3].astype(np.float64), axis=1)
    holder = [raw]
    del raw
    est = main.estimate(ts, holder)
    main.write_output(out, ts, est)
    runtime = time.perf_counter() - t0
    st = est["stats"]
    if st.get("column_order") == "accel,gyro":
        _, raw2 = main.read_imu(imu)
        gyro_norm = np.linalg.norm(raw2[:, 3:6].astype(np.float64), axis=1)

    t_sec = (ts - ts[0]) * 1e-9
    dur = float(t_sec[-1])
    te, pe, qe = evaluate.load_est(out)
    data = {"title": title or os.path.basename(imu), "duration": dur, "rows": int(len(ts)),
            "runtime_ms_per_s": 1000 * runtime / max(dur, 1e-9), "stats": {k: (round(v, 3) if isinstance(v, float) else v)
                                                                            for k, v in st.items()}}
    lv = est["last_valid"]
    data["blind_start"] = float(t_sec[lv]) if st["blind_rows"] else None

    idx = downsample(len(te))
    if gt:
        tg, pg, _ = evaluate.load_gt(gt)
        r = evaluate.evaluate(out, gt, imu)
        _, first = np.unique(te, return_index=True)
        te_u, pe_u = te[first], pe[first]
        j = np.clip(np.searchsorted(tg, te_u), 1, len(tg) - 1)
        ok = np.abs(tg[j] - te_u) < 0.01
        R, tr = evaluate.umeyama_se3(pe_u[ok], pg[j[ok]])
        pe_al = pe @ R.T + tr
        jj = np.clip(np.searchsorted(tg, te), 0, len(tg) - 1)
        gtp = pg[jj]
        # naive baseline for the comparison bar
        nv = os.path.join(tmp, "naive.txt")
        main.run(imu, nv, p=dict(main.PARAMS, kp=0.0, ki=0.0, spike_thresh=1e9, hp_cut_hz=0.0, vel_hp=False))
        rn = evaluate.evaluate(nv, gt, imu)
        data["scores"] = {"ate": r["ATE_rmse_m"], "floor": r["GT_spread_m"], "naive": rn["ATE_rmse_m"],
                          "blind": r["T2_blind5s_ATE_m"], "waypoint": r["T1_waypoint_ATE_m"],
                          "rpe_rot": r["RPE_rot_1s_deg"]}
        data["gt"] = np.round(gtp[idx], 3).tolist()
        path = pe_al
    else:
        path = pe
    data["t"] = np.round(te[idx] - te[0], 3).tolist()
    data["path"] = np.round(path[idx], 3).tolist()
    data["quat"] = np.round(qe[idx], 4).tolist()

    # forensics timeline
    dts = np.diff(t_sec)
    nominal = float(np.median(dts))
    gaps = np.flatnonzero(dts > 2.5 * nominal)
    data["gaps"] = [[round(float(t_sec[g]), 3), round(float(dts[g]), 3)] for g in gaps]
    data["dups"] = [round(float(t_sec[i + 1]), 3) for i in np.flatnonzero(dts == 0)]
    tu = (est["t_ns"] - est["t_ns"][0]) * 1e-9
    ev = est.get("events", {})
    data["stuck"] = [round(float(tu[i]), 3) for i in ev.get("stuck", [])]
    data["spikes"] = [round(float(tu[i]), 3) for i in ev.get("spike", [])]
    k = downsample(len(t_sec), 1200)
    w = max(1, len(t_sec) // 1200)
    gsm = np.convolve(gyro_norm, np.ones(w) / w, mode="same")
    data["gyro"] = [[round(float(t_sec[i]), 2), round(float(gsm[i]), 3)] for i in k]
    return data


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Driftlock mission report</title>
<style>
:root{--bg:#070b16;--panel:#0e1529;--panel2:#121c36;--line:#1f2a48;--text:#e8edf7;--muted:#8c99b8;--cyan:#22d3ee;--teal:#2dd4bf;--orange:#fb923c;--red:#f43f5e;--gt:#cbd5e1}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#13254d 0%,var(--bg) 60%);color:var(--text);font-family:Inter,system-ui,sans-serif;min-height:100vh}
.wrap{max-width:1800px;margin:0 auto;padding:22px 28px 28px}
header{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.logo{display:flex;align-items:center;gap:12px;font:800 30px Poppins,Inter,sans-serif;letter-spacing:-.5px}
.logo svg{filter:drop-shadow(0 0 12px rgba(34,211,238,.55))}
.sub{color:var(--muted);font-size:15px}.chips{margin-left:auto;display:flex;gap:10px;flex-wrap:wrap}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:8px 14px;font-size:14px;color:var(--muted)}
.chip b{color:var(--text);font-weight:600}.chip.ok b{color:var(--teal)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:18px;margin-top:18px;align-items:start}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:18px;padding:16px 18px;position:relative}
.card h3{margin:0 0 10px;font:600 13px Poppins,Inter;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
#view{width:100%;height:600px;display:block;cursor:grab;border-radius:12px}
.legend{position:absolute;left:30px;top:52px;display:flex;flex-direction:column;gap:6px;font-size:13px;color:var(--muted);pointer-events:none}
.legend span{display:flex;align-items:center;gap:8px}.legend i{width:22px;height:4px;border-radius:2px;display:inline-block}
.hud{position:absolute;right:30px;top:52px;text-align:right;pointer-events:none}
.hud .big{font:700 34px Poppins,Inter;line-height:1}.hud .small{color:var(--muted);font-size:13px;margin-top:4px}
.blindtag{display:none;margin-top:10px;background:rgba(251,146,60,.14);color:var(--orange);border:1px solid rgba(251,146,60,.5);padding:6px 10px;border-radius:8px;font-weight:600;font-size:14px}
.controls{display:flex;align-items:center;gap:12px;margin-top:10px}
button{background:var(--cyan);color:#04121a;border:0;border-radius:10px;padding:9px 16px;font:700 14px Inter;cursor:pointer}
button.ghost{background:transparent;color:var(--text);border:1px solid var(--line)}
input[type=range]{flex:1;accent-color:var(--cyan)}
.stat{display:flex;justify-content:space-between;align-items:baseline;padding:10px 0;border-bottom:1px solid var(--line)}
.stat:last-child{border-bottom:0}.stat .k{color:var(--muted);font-size:14px}.stat .v{font:700 22px Poppins,Inter}
.bars{display:flex;flex-direction:column;gap:12px;margin-top:4px}
.bar{font-size:13px;color:var(--muted)}.bar .track{height:12px;background:#0a1122;border-radius:6px;margin-top:5px;overflow:hidden}
.bar .fill{height:100%;border-radius:6px}.bar b{color:var(--text);float:right}
.f{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.fx{background:#0a1122;border:1px solid var(--line);border-radius:12px;padding:10px}
.fx .n{font:700 24px Poppins,Inter}.fx .l{font-size:12px;color:var(--muted);margin-top:2px}
#tl{width:100%;height:160px;display:block}
.foot{color:var(--muted);font-size:12px;margin-top:14px}
@media (max-width:1100px){.grid{grid-template-columns:1fr}#view{height:420px}}
</style></head><body><div class="wrap">
<header><div class="logo"><svg width="40" height="40" viewBox="0 0 48 48"><path d="M6 36 C14 10, 30 40, 42 12" stroke="#22d3ee" stroke-width="4" fill="none" stroke-linecap="round"/><circle cx="42" cy="12" r="5" fill="#fb923c"/><circle cx="6" cy="36" r="4" fill="#2dd4bf"/></svg>Driftlock</div>
<div class="sub" id="title"></div><div class="chips" id="chips"></div></header>
<div class="grid"><div class="card"><h3>Flight path &middot; drag to rotate</h3><canvas id="view"></canvas>
<div class="legend" id="legend"></div><div class="hud"><div class="big" id="clock">0.0 s</div><div class="small" id="pos"></div><div class="blindtag" id="blindtag">IMU DEAD &middot; dead reckoning</div></div>
<div class="controls"><button id="play">Pause</button><button class="ghost" id="speed">4&times;</button><input type="range" id="scrub" min="0" max="1000" value="0"></div></div>
<div style="display:flex;flex-direction:column;gap:18px">
<div class="card" id="scorecard"><h3>Score vs hidden ground truth</h3><div id="scores"></div></div>
<div class="card"><h3>Forensics &middot; what the jamming did</h3><div class="f" id="forensics"></div></div>
<div class="card"><h3>Budgets</h3><div id="budget"></div></div></div></div>
<div class="card" style="margin-top:18px"><h3>Sensor timeline &middot; detected damage</h3><canvas id="tl"></canvas></div>
<div class="foot">Generated by tools/report.py &middot; Samsung XR Hackathon 2026, Track 03 &middot; all numbers computed from this run</div>
</div><script>
const D=__DATA__;
const $=id=>document.getElementById(id);
$('title').textContent='Mission report · '+D.title;
const s=D.stats, fmt=(x,d=2)=>Number(x).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});
$('chips').innerHTML=`<div class="chip"><b>${D.rows.toLocaleString()}</b> IMU rows</div><div class="chip"><b>${fmt(D.duration,1)} s</b> flight</div><div class="chip ok"><b>${D.rows.toLocaleString()}</b> poses out</div><div class="chip ok"><b>${fmt(D.runtime_ms_per_s,1)} ms</b> per IMU-second</div>`;
const L_=x=>Number(x).toLocaleString();const F=[[L_(s.blind_rows),'dead-sensor rows',"#fb923c"],[s.gaps,'time gaps',"#22d3ee"],[fmt(s.gap_seconds,1)+' s','missing time',"#22d3ee"],[s.duplicate_rows,'duplicate rows',"#a78bfa"],[s.stuck_rows,'frozen rows',"#2dd4bf"],[s.spike_rows,'spikes removed',"#f43f5e"]];
$('forensics').innerHTML=F.map(([n,l,c])=>`<div class="fx"><div class="n" style="color:${c}">${n}</div><div class="l">${l}</div></div>`).join('');
$('budget').innerHTML=`<div class="stat"><span class="k">Compute per IMU-second</span><span class="v" style="color:var(--teal)">${fmt(D.runtime_ms_per_s,1)} <small style="font-size:13px;color:var(--muted)">/ 20 ms</small></span></div><div class="stat"><span class="k">Threads</span><span class="v">1</span></div><div class="stat"><span class="k">Column order found</span><span class="v" style="font-size:16px">${s.column_order}</span></div>`;
if(D.scores){const S=D.scores,mx=Math.log10(S.naive*1.2),w=v=>Math.max(3,100*Math.log10(Math.max(v,1))/mx+8);
$('scores').innerHTML=`<div class="stat"><span class="k">Driftlock ATE</span><span class="v" style="color:var(--teal)">${fmt(S.ate)} m</span></div><div class="stat"><span class="k">Blind 5 s ATE</span><span class="v" style="color:var(--orange)">${fmt(S.blind)} m</span></div>
<div class="bars"><div class="bar">Driftlock<b>${fmt(S.ate)} m</b><div class="track"><div class="fill" style="width:${w(S.ate)}%;background:var(--teal)"></div></div></div>
<div class="bar">Do nothing (stay at start)<b>${fmt(S.floor)} m</b><div class="track"><div class="fill" style="width:${w(S.floor)}%;background:#64748b"></div></div></div>
<div class="bar">Textbook integration<b>${Math.round(S.naive).toLocaleString()} m</b><div class="track"><div class="fill" style="width:100%;background:var(--red)"></div></div></div></div>
<div class="foot" style="margin-top:8px">Log scale. SE(3)-aligned, same metric as evo.</div>`}else{$('scorecard').style.display='none'}
$('legend').innerHTML=(D.gt?'<span><i style="background:#cbd5e1"></i>Ground truth (hidden from the estimator)</span>':'')+'<span><i style="background:#22d3ee"></i>Driftlock estimate</span><span><i style="background:#fb923c"></i>Blind 5 s (IMU dead)</span>';
// ---------- 3D view
const cv=$('view'),cx=cv.getContext('2d');let W,H,dpr=window.devicePixelRatio||1;
function size(){W=cv.clientWidth;H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;cx.setTransform(dpr,0,0,dpr,0,0)}size();addEventListener('resize',()=>{size();draw()});
const P=D.path,G=D.gt,T=D.t,N=P.length;const all=G?P.concat(G):P;
const c=[0,1,2].map(k=>{let a=Infinity,b=-Infinity;all.forEach(p=>{a=Math.min(a,p[k]);b=Math.max(b,p[k])});return [(a+b)/2,b-a]});
const span=Math.max(c[0][1],c[1][1],c[2][1]*1.5,0.5);
let yaw=-0.7,pitch=0.55,drag=null,auto=true;
cv.onmousedown=e=>{drag=[e.clientX,e.clientY];auto=false;cv.style.cursor='grabbing'};onmouseup=()=>{drag=null;cv.style.cursor='grab'};
onmousemove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*0.008));drag=[e.clientX,e.clientY];draw()};
function pr(p){let x=(p[0]-c[0][0])/span,y=(p[1]-c[1][0])/span,z=(p[2]-c[2][0])/span;
 let X=x*Math.cos(yaw)-y*Math.sin(yaw),Y=x*Math.sin(yaw)+y*Math.cos(yaw);let Z=z*Math.cos(pitch)-Y*Math.sin(pitch),YY=z*Math.sin(pitch)+Y*Math.cos(pitch);
 const f=2.6/(2.6+YY),sc=Math.min(W,H)*1.15;return [W/2+X*sc*f,H/2-Z*sc*f]}
function grid(){cx.strokeStyle='rgba(148,163,184,.10)';cx.lineWidth=1;const zf=c[2][0]-c[2][1]/2;
 for(let i=-5;i<=5;i++){const a=i/10*span*1.2;let p1=pr([c[0][0]+a,c[1][0]-span*0.6,zf]),p2=pr([c[0][0]+a,c[1][0]+span*0.6,zf]);cx.beginPath();cx.moveTo(...p1);cx.lineTo(...p2);cx.stroke();
 p1=pr([c[0][0]-span*0.6,c[1][0]+a,zf]);p2=pr([c[0][0]+span*0.6,c[1][0]+a,zf]);cx.beginPath();cx.moveTo(...p1);cx.lineTo(...p2);cx.stroke()}}
let k=0,playing=true,speed=4,last=performance.now(),tcur=0;
function line(arr,upto,col,wid,from=0,alpha=1){if(upto-from<1)return;cx.globalAlpha=alpha;cx.strokeStyle=col;cx.lineWidth=wid;cx.lineJoin='round';cx.beginPath();let q=pr(arr[from]);cx.moveTo(...q);for(let i=from+1;i<=upto;i++){q=pr(arr[i]);cx.lineTo(...q)}cx.stroke();cx.globalAlpha=1}
const bi=D.blind_start==null?N:T.findIndex(t=>t>=D.blind_start);
function draw(){cx.clearRect(0,0,W,H);grid();
 if(G){line(G,N-1,'#cbd5e1',1.4,0,0.18);line(G,k,'#cbd5e1',2,0,0.75)}
 line(P,N-1,'#22d3ee',1.2,0,0.12);
 line(P,Math.min(k,bi>0?bi:N-1),'#22d3ee',2.6);if(k>bi&&bi>0)line(P,k,'#fb923c',3.4,bi);
 const h=pr(P[k]);cx.shadowColor=k>bi?'#fb923c':'#22d3ee';cx.shadowBlur=22;cx.fillStyle=k>bi?'#fb923c':'#e0faff';cx.beginPath();cx.arc(h[0],h[1],8,0,7);cx.fill();cx.shadowBlur=0;
 if(G){const g=pr(G[k]);cx.strokeStyle='#cbd5e1';cx.lineWidth=2;cx.beginPath();cx.arc(g[0],g[1],7,0,7);cx.stroke()}
 $('clock').textContent=T[k].toFixed(1)+' s';$('pos').textContent=`x ${P[k][0].toFixed(2)}  y ${P[k][1].toFixed(2)}  z ${P[k][2].toFixed(2)} m`;
 $('blindtag').style.display=k>bi?'inline-block':'none';$('scrub').value=Math.round(1000*k/(N-1));drawTL()}
function tick(now){const dt=(now-last)/1000;last=now;if(playing){tcur+=dt*speed;if(tcur>T[N-1]+2)tcur=0;while(k<N-1&&T[k+1]<=tcur)k++;if(tcur<T[k])k=0;}if(auto)yaw+=dt*0.12;draw();requestAnimationFrame(tick)}
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play'};
$('speed').onclick=()=>{speed=speed==4?10:speed==10?1:4;$('speed').innerHTML=speed+'&times;'};
$('scrub').oninput=e=>{k=Math.round(e.target.value/1000*(N-1));tcur=T[k]};
// ---------- timeline
const tl=$('tl'),tx=tl.getContext('2d');function tsize(){tl.width=tl.clientWidth*dpr;tl.height=tl.clientHeight*dpr;tx.setTransform(dpr,0,0,dpr,0,0)}tsize();addEventListener('resize',tsize);
function drawTL(){const w=tl.clientWidth,h=tl.clientHeight,L=50,R=w-14,X=t=>L+(R-L)*t/D.duration;tx.clearRect(0,0,w,h);
 const gmax=Math.max(...D.gyro.map(g=>g[1]))*1.1;tx.fillStyle='#8c99b8';tx.font='12px Inter';
 tx.fillText('motion',4,40);tx.fillText('gaps',4,92);tx.fillText('rows',4,122);
 tx.strokeStyle='rgba(148,163,184,.75)';tx.lineWidth=1.2;tx.beginPath();D.gyro.forEach((g,i)=>{const x=X(g[0]),y=72-62*g[1]/gmax;i?tx.lineTo(x,y):tx.moveTo(x,y)});tx.stroke();
 if(D.blind_start!=null){tx.fillStyle='rgba(251,146,60,.18)';tx.fillRect(X(D.blind_start),4,R-X(D.blind_start),134);tx.fillStyle='#fb923c';tx.font='600 12px Inter';tx.fillText('IMU DEAD',X(D.blind_start)-70,20)}
 D.gaps.forEach(([t,d])=>{tx.fillStyle='#22d3ee';tx.fillRect(X(t),97-Math.min(20,6+10*d),Math.max(2,X(t+d)-X(t)),Math.min(20,6+10*d))});
 const dot=(arr,col,y)=>{tx.fillStyle=col;arr.forEach(t=>{tx.fillRect(X(t)-1,y,2,9)})};dot(D.dups,'#a78bfa',104);dot(D.stuck,'#2dd4bf',115);dot(D.spikes,'#f43f5e',126);
 tx.font='12px Inter';let lx=L;[['gap','#22d3ee'],['duplicate','#a78bfa'],['frozen','#2dd4bf'],['spike','#f43f5e'],['IMU dead','#fb923c']].forEach(([n,c])=>{tx.fillStyle=c;tx.fillRect(lx,h-12,10,10);tx.fillStyle='#8c99b8';tx.fillText(n,lx+15,h-3);lx+=tx.measureText(n).width+38});
 const px=X(T[k]);tx.strokeStyle='#e8edf7';tx.lineWidth=1.5;tx.beginPath();tx.moveTo(px,4);tx.lineTo(px,h-4);tx.stroke()}
requestAnimationFrame(tick);
</script></body></html>"""


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("imu")
    ap.add_argument("--gt")
    ap.add_argument("-o", "--out", default="report.html")
    ap.add_argument("--title")
    a = ap.parse_args()
    data = build(a.imu, a.gt, a.title)
    with open(a.out, "w") as f:
        f.write(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print("wrote", a.out)


if __name__ == "__main__":
    main_()
