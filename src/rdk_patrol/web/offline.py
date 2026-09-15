from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from rdk_patrol.alarms import AlarmRepository
from rdk_patrol.atomic_io import atomic_write_text


_STYLE = """
:root{color-scheme:light;font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#17202a}
header{background:#17324d;color:#fff;padding:20px 24px}
h1{font-size:22px;margin:0 0 6px}.meta{opacity:.8;font-size:13px}
.toolbar{display:flex;gap:12px;flex-wrap:wrap;padding:14px 24px;background:#fff;border-bottom:1px solid #dce2e8;position:sticky;top:0;z-index:2}
label{font-size:13px;color:#52616f}select,input{margin-left:6px;padding:7px;border:1px solid #aeb8c2;border-radius:5px}
main{padding:20px 24px}.empty{padding:40px;text-align:center;color:#687784}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
article{background:#fff;border:1px solid #dce2e8;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px #1c2a3512}
article img{width:100%;height:220px;object-fit:contain;background:#111;display:block}
dl{display:grid;grid-template-columns:90px 1fr;margin:0;padding:14px 16px;gap:8px 10px;font-size:14px}
dt{font-weight:600;color:#52616f}dd{margin:0;overflow-wrap:anywhere}
.event{display:inline-block;padding:3px 8px;border-radius:999px;background:#ffe8e8;color:#a20d0d;font-weight:700}
.health{margin:16px 24px 0;padding:14px 16px;background:#fff;border:1px solid #dce2e8;border-radius:8px}
.health-head{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.health strong{font-size:16px}
.health-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.health-item{padding:9px;border-radius:6px;background:#f2f5f8;font-size:13px}.health-item b{display:block;margin-bottom:3px}
.state-ok{color:#087a38}.state-degraded,.state-starting{color:#9a6200}.state-failed,.state-stopped{color:#b00020}
@media(max-width:600px){.toolbar,main{padding-left:12px;padding-right:12px}.grid{grid-template-columns:1fr}}
"""


_FILTER_SCRIPT = """
(function(){
  const cards=[...document.querySelectorAll("article[data-event]")];
  const eventSelect=document.getElementById("event-filter");
  const pointInput=document.getElementById("point-filter");
  function apply(){
    const event=eventSelect.value, point=pointInput.value.trim().toLowerCase();
    let shown=0;
    cards.forEach(card=>{
      const okEvent=!event||card.dataset.event===event;
      const okPoint=!point||(card.dataset.point||"").toLowerCase().includes(point);
      card.hidden=!(okEvent&&okPoint); if(!card.hidden)shown++;
    });
    document.getElementById("shown-count").textContent=String(shown);
  }
  eventSelect.addEventListener("change",apply);
  pointInput.addEventListener("input",apply);
  apply();
})();
"""


def _image_data_uri(repository: AlarmRepository, relative_path: str) -> str:
    path = repository.resolve_image(relative_path)
    if path is None:
        return ""
    payload = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return "data:{};base64,{}".format(mime, base64.b64encode(payload).decode("ascii"))


def build_offline_html(
    repository: AlarmRepository,
    limit: Optional[int] = None,
) -> str:
    """Build one portable HTML file with all selected images embedded."""

    records = repository.list_records(limit=limit, newest_first=True)
    events = sorted({str(record["event_name"]) for record in records})
    options = ['<option value="">全部事件</option>']
    options.extend(
        '<option value="{0}">{0}</option>'.format(html.escape(value, quote=True))
        for value in events
    )
    cards: List[str] = []
    for record in records:
        image_uri = _image_data_uri(repository, record["keyframe_image"])
        image_markup = (
            '<img alt="报警关键帧" src="{}">'.format(image_uri)
            if image_uri
            else '<div class="empty">关键帧不可用</div>'
        )
        point = record.get("point_name", "")
        point_row = (
            "<dt>点位名称</dt><dd>{}</dd>".format(html.escape(point))
            if point
            else ""
        )
        cards.append(
            (
                '<article data-event="{event_attr}" data-point="{point_attr}">'
                "{image}<dl>"
                '<dt>事件名称</dt><dd><span class="event">{event}</span></dd>'
                "<dt>北京时间</dt><dd>{time}</dd>"
                "{point_row}"
                "<dt>关键帧</dt><dd>{path}</dd>"
                "</dl></article>"
            ).format(
                event_attr=html.escape(record["event_name"], quote=True),
                point_attr=html.escape(point, quote=True),
                image=image_markup,
                event=html.escape(record["event_name"]),
                time=html.escape(record["beijing_time"]),
                point_row=point_row,
                path=html.escape(record["keyframe_image"]),
            )
        )
    content = "\n".join(cards) if cards else '<div class="empty">暂无报警记录</div>'
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RDK S100 巡检报警台账</title><style>{style}</style></head>
<body><header><h1>RDK S100 巡检报警台账</h1>
<div class="meta">离线自包含文件 · 共 <span id="shown-count">{count}</span> 条</div></header>
<section class="toolbar"><label>事件<select id="event-filter">{options}</select></label>
<label>点位<input id="point-filter" placeholder="输入点位名称"></label></section>
<main><div class="grid">{content}</div></main>
<script>{script}</script></body></html>
""".format(
        style=_STYLE,
        count=len(records),
        options="".join(options),
        content=content,
        script=_FILTER_SCRIPT,
    )


def export_offline_html(
    destination: Union[str, Path],
    repository: AlarmRepository,
    limit: Optional[int] = None,
) -> Path:
    path = Path(destination)
    atomic_write_text(path, build_offline_html(repository, limit=limit))
    return path


def build_live_dashboard_html() -> str:
    """Small dependency-free dashboard that polls only read endpoints."""

    script = r"""
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const wanted=[["camera","相机"],["inference","推理"],["localization","点位定位"],["stereo","双目测距"],["recording","录像"],["minimax_queue","MiniMax队列"]];
function component(h,key){
  const all=h.components||{};
  if(all[key])return all[key];
  if(key==="minimax_queue")return all.garbage_review||all.minimax||all.review_queue||null;
  return null;
}
function metricFps(h){
  const m=h.metrics||{};
  if(m.measured_pipeline_fps_5s!=null)return m.measured_pipeline_fps_5s;
  if(m.measured_fps!=null)return m.measured_fps;if(m.fps!=null)return m.fps;
  for(const key of ["pipeline","inference","camera"]){const c=component(h,key);if(c&&c.metrics){if(c.metrics.fps_5s!=null)return c.metrics.fps_5s;if(c.metrics.fps!=null)return c.metrics.fps;if(c.metrics.measured_fps!=null)return c.metrics.measured_fps;}}
  return null;
}
function renderHealth(h){
  const state=String(h.status||"unknown");
  const fps=metricFps(h), target=(h.metrics||{}).target_fps??15;
  document.getElementById("health-overall").innerHTML=`总体：<span class="state-${esc(state)}">${esc(state)}</span>`;
  document.getElementById("health-fps").textContent=`实测 FPS：${fps==null?"未上报":Number(fps).toFixed(1)} / 目标 ${target}`;
  document.getElementById("health-components").innerHTML=wanted.map(([key,label])=>{const c=component(h,key);const s=c?String(c.state||"unknown"):"未上报";let detail=c&&c.message?String(c.message):"";
    if(key==="minimax_queue"&&c&&c.metrics){const n=c.metrics.pending_queue??c.metrics.pending_count??c.metrics.queue_size;if(n!=null)detail=`待处理 ${n} 条${detail?" · "+detail:""}`;}
    return `<div class="health-item"><b>${esc(label)}</b><span class="state-${esc(s)}">${esc(s)}</span>${detail?`<div>${esc(detail)}</div>`:""}</div>`;}).join("");
}
async function refresh(){
  const event=document.getElementById("event-filter").value;
  const point=document.getElementById("point-filter").value.trim();
  const q=new URLSearchParams({limit:"200"});if(event)q.set("event_name",event);if(point)q.set("point_name",point);
  try{
    const [response,healthResponse]=await Promise.all([fetch("/api/alarms?"+q.toString(),{cache:"no-store"}),fetch("/health",{cache:"no-store"})]);
    const data=await response.json();const health=await healthResponse.json();const rows=data.alarms||[];renderHealth(health);
    document.getElementById("shown-count").textContent=String(rows.length);
    const events=[...new Set(rows.map(x=>x.event_name))].sort();
    const select=document.getElementById("event-filter");
    const current=select.value;
    if(!select.dataset.ready){select.innerHTML='<option value="">全部事件</option>'+events.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join("");select.value=current;select.dataset.ready="1";}
    document.querySelector(".grid").innerHTML=rows.length?rows.map(r=>`<article data-event="${esc(r.event_name)}" data-point="${esc(r.point_name||"")}"><img alt="报警关键帧" src="/${encodeURI(r.keyframe_image)}?v=${encodeURIComponent(r.beijing_time)}"><dl><dt>事件名称</dt><dd><span class="event">${esc(r.event_name)}</span></dd><dt>北京时间</dt><dd>${esc(r.beijing_time)}</dd>${r.point_name?`<dt>点位名称</dt><dd>${esc(r.point_name)}</dd>`:""}<dt>关键帧</dt><dd>${esc(r.keyframe_image)}</dd></dl></article>`).join(""):'<div class="empty">暂无报警记录</div>';
    document.getElementById("status").textContent="已连接 · "+new Date().toLocaleTimeString();
  }catch(e){document.getElementById("status").textContent="连接失败，正在重试";}
}
document.getElementById("event-filter").addEventListener("change",refresh);
document.getElementById("point-filter").addEventListener("input",()=>{clearTimeout(window._ft);window._ft=setTimeout(refresh,250)});
refresh();setInterval(refresh,5000);
"""
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RDK S100 实时报警</title><style>{style}</style></head>
<body><header><h1>RDK S100 实时报警</h1>
<div class="meta"><span id="status">正在连接</span> · 当前 <span id="shown-count">0</span> 条</div></header>
<section class="toolbar"><label>事件<select id="event-filter"><option value="">全部事件</option></select></label>
<label>点位<input id="point-filter" placeholder="精确点位名称"></label>
<a href="/health" target="_blank" rel="noopener">查看健康 JSON</a>
<a href="/download/offline.html">下载离线台账</a>
<a href="/download/alarms.jsonl">下载 JSONL</a></section>
<section class="health"><div class="health-head"><strong id="health-overall">总体：正在加载</strong>
<span id="health-fps">实测 FPS：正在加载 / 目标 15</span></div>
<div class="health-grid" id="health-components"></div></section>
<main><div class="grid"><div class="empty">正在加载……</div></div></main>
<script>{script}</script></body></html>
""".format(style=_STYLE, script=script)
