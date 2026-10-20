"""Self-contained interactive report; kept with ignored match-derived outputs."""

import html
import json
from pathlib import Path

import numpy as np


def write_report(path, labels, posterior, histories, disagreement, arrays, queries, scores, manifest, notice=""):
    stride = 25
    codes = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[x] for x in labels])
    seen = np.isfinite(arrays["ball"]).all(1)
    payload = {"step": stride, "n": len(labels), "labels": codes[::stride].tolist(),
               "p": np.round(posterior[::stride], 3).tolist(),
               "seen": seen[::stride].astype(int).tolist(),
               "tag": np.round(arrays["tag"][::stride], 2).tolist(),
               "disagreement": np.round(disagreement[::stride], 3).tolist(),
               "histories": histories[:6, ::stride].tolist(), "queries": queries, "scores": scores}
    template = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BlindSpot — possession evidence</title><style>
body{margin:0;background:#0a1220;color:#e6edf7;font:15px system-ui}main{max-width:1180px;margin:auto;padding:32px}h1{font-size:40px;margin:5px 0}.muted{color:#a2b4ce}.badge{background:#1a304b;padding:6px 12px;border-radius:20px;display:inline-block}section{background:#111f32;border:1px solid #263a53;border-radius:14px;padding:22px;margin:20px 0}button{background:#234e6d;border:0;color:white;padding:9px 14px;border-radius:8px;cursor:pointer;margin:4px}canvas{width:100%;height:310px;background:#0c1726;border-radius:10px}input{width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #263a53}tr.query{cursor:pointer}tr.query:hover{background:#213550}.cols{display:flex;gap:18px;flex-wrap:wrap}.legend span{margin-right:18px}code{color:#6ee7cc}a{color:#6ee7cc}</style>
<main><span class="badge">Research prototype · local evidence</span><h1>BlindSpot</h1><p class="muted">Possession under missing evidence. Alternative histories, source sensitivity, and budgeted video queries.</p><p class="muted">__NOTICE__</p>
<section><div class="cols"><div><b>Home</b> Barcelona</div><div><b>Away</b> Athletic Club</div><div><b>Frames</b> 199,319 · 25 fps</div></div><p class="muted">Model support is not calibrated confidence. Agreement among sampled histories is not a correctness certificate. Public evaluation covers five minutes; no private-match score is known.</p></section>
<section><h2>Evidence timeline</h2><div class="legend"><span style="color:#51c5ff">■ Home</span><span style="color:#c9a1ff">■ Away</span><span style="color:#ffba6b">■ Dead</span><span style="color:#60728a">■ Ungraded</span></div><p id="window"></p><canvas id="chart" width="1100" height="310"></canvas><input id="position" type="range" min="0" max="191819" value="11185"><button onclick="jump(11185)">First half</button><button onclick="jump(110808)">Second half</button><button onclick="nextQuery()">Next ambiguous moment</button><p class="muted">Rows: final prediction · sampled alternative histories · ball observed · source disagreement · model uncertainty. Click the chart to inspect a frame.</p><p id="inspect"></p></section>
<section><h2>Public-window experiments</h2><div id="scores"></div><p class="muted">Ablations use fixed settings. Calibration-trained scores must be distinguished from held-out scores. More possessions or higher agreement are not automatically better.</p></section>
<section><h2>Where should the system look next?</h2><p class="muted">Queries are chosen without reference labels. They mark uncertainty, not known errors. Visual motion does not identify the possessor.</p><div id="queries"></div></section>
<section><h2>Provenance</h2><pre style="white-space:pre-wrap">__MANIFEST__</pre></section></main>
<script>const D=__PAYLOAD__;const C=['#51c5ff','#c9a1ff','#ffba6b','#26354a'];let q=0;const slider=document.getElementById('position');const chart=document.getElementById('chart');const ctx=chart.getContext('2d');let width=7500;
function time(f){return (f/25/60).toFixed(2)+' min'}function jump(f){slider.value=Math.min(+slider.max,Math.max(0,f-500));draw()}function nextQuery(){if(D.queries.length)jump(D.queries[q++%D.queries.length].frame)}
function draw(){let start=+slider.value;ctx.clearRect(0,0,1100,310);document.getElementById('window').textContent='Video frames '+start+'–'+Math.min(D.n-1,start+width)+' · '+time(start);let rows=[D.labels,...D.histories];for(let x=0;x<1100;x++){let i=Math.floor((start+x*width/1100)/D.step);rows.forEach((r,k)=>{ctx.fillStyle=C[r[i]??3];ctx.fillRect(x,k*20,1,15)});let y=rows.length*20;ctx.fillStyle=D.seen[i]?'#70dfba':'#26354a';ctx.fillRect(x,y,1,13);ctx.fillStyle='#ff7f91';ctx.fillRect(x,y+20,1,35*(D.disagreement[i]||0));let p=D.p[i]||[0,0,0];let entropy=-p.reduce((s,v)=>s+(v? v*Math.log(v):0),0)/Math.log(3);ctx.fillStyle='#e3d474';ctx.fillRect(x,y+65,1,35*entropy)}}slider.oninput=draw;chart.onclick=e=>{let f=Math.min(D.n-1,Math.floor(+slider.value+(e.offsetX/chart.clientWidth)*width));let i=Math.floor(f/D.step);document.getElementById('inspect').textContent='Frame '+f+' · '+time(f)+' · State '+['H','A','D','X'][D.labels[i]]+' · model support '+JSON.stringify(D.p[i])+' · ball '+(D.seen[i]?'observed':'missing')};
let rows=Object.entries(D.scores).map(([name,r])=>'<tr><td>'+name+'</td><td>'+r.macro_f1.toFixed(4)+'</td><td>'+r.spell_f1_mean.toFixed(4)+'</td><td>'+r.n_pred_spells+'</td><td>'+r.scope+'</td></tr>').join('');document.getElementById('scores').innerHTML='<table><tr><th>Experiment</th><th>macroF1</th><th>spellF1</th><th>Possessions</th><th>Evaluation scope</th></tr>'+rows+'</table>';
document.getElementById('queries').innerHTML='<table><tr><th>Frame</th><th>Half</th><th>Score</th><th>Ball</th></tr>'+D.queries.map(r=>'<tr class="query" onclick="jump('+r.frame+')"><td>'+r.frame+'</td><td>'+r.half+'</td><td>'+r.score.toFixed(3)+'</td><td>'+(r.ball_missing?'missing':'observed')+'</td></tr>').join('')+'</table>';draw();</script></html>'''
    manifest_text = html.escape(json.dumps(manifest, indent=2))
    rendered = template.replace("__MANIFEST__", manifest_text).replace("__NOTICE__", html.escape(notice)).replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/"))
    Path(path).write_text(rendered, encoding="utf-8")
