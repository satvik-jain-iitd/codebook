from flask import Flask, jsonify, request, current_app, send_from_directory
from pathlib import Path
import sqlite3
import os

app = Flask(__name__)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "static"), filename
    )

@app.route("/graph")
def graph_view():
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Codebase Knowledge Graph</title>
<style>
  body { margin: 0; font-family: sans-serif; overflow: hidden; }
  #legend { position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 8px 14px; border-radius: 6px; font-size: 12px; }
  #legend span { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 4px; }
  .link { stroke: #999; stroke-opacity: 0.5; }
  .link.CALLS { stroke: #4caf50; }
  .link.IMPORTS { stroke: #2196f3; }
  .link.CONTAINS { stroke: #ff9800; }
</style>
</head>
<body>
<div id="legend">
  <div><span style="background:#4caf50"></span> CALLS</div>
  <div><span style="background:#2196f3"></span> IMPORTS</div>
  <div><span style="background:#ff9800"></span> CONTAINS</div>
</div>
<div id="graph"></div>
<script src="/static/d3.min.js"></script>
<script>
const W = window.innerWidth, H = window.innerHeight;
const svg = d3.select("#graph").append("svg").attr("width", W).attr("height", H);
const defs = svg.append("defs");
defs.append("marker").attr("id", "arrow").attr("viewBox", "0 -5 10 10").attr("refX", 20).attr("refY", 0)
  .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#999");

Promise.all([
  fetch("/api/symbols?limit=200").then(r => r.json()),
  fetch("/api/relations").then(r => r.json())
]).then(([data, relations]) => {
  const symbols = data.symbols;
  const total = data.total;
  if (total > symbols.length) {
    d3.select("#legend").append("p").attr("style", "color:red;font-weight:bold;margin-top:8px")
      .text(`⚠️ Showing ${symbols.length} of ${total} symbols`);
  }
  const nodes = symbols.map(s => ({id: s.id, name: s.name}));
  const nodeIds = new Set(nodes.map(n => n.id));
  const links = relations.filter(r => nodeIds.has(r.from_id) && nodeIds.has(r.to_id))
    .map(r => ({ source: r.from_id, target: r.to_id, type: r.type }));
  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(120))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(W/2, H/2));
  const link = svg.selectAll(".link").data(links).join("line")
    .attr("class", d => `link ${d.type}`)
    .attr("marker-end", "url(#arrow)");
  const node = svg.selectAll(".node").data(nodes).join("g").attr("class", "node")
    .call(d3.drag().on("start", (e,d) => { if(!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (e,d) => { d.fx = e.x; d.fy = e.y; })
      .on("end", (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));
  node.append("circle").attr("r", 7).attr("fill", "#69b3a2").attr("stroke", "#333").attr("stroke-width", 1.5);
  node.append("text").attr("dx", 12).attr("dy", 4).style("font-size", "11px").text(d => d.name);
  sim.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
  });
});
</script>
</body></html>"""

def get_db_connection():
    db_path = current_app.config.get('DB_PATH')
    if not db_path:
        db_path = Path(".codebook_cache.db").resolve()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    db_path = current_app.config.get('DB_PATH', 'Not set (using default)')
    rows_html = ""
    try:
        conn = get_db_connection()
        relations = conn.execute("""
            SELECT s1.name as source_name, r.type, s2.name as target_name
            FROM relations r
            JOIN symbols s1 ON r.source_id = s1.id
            JOIN symbols s2 ON r.target_id = s2.id
            LIMIT 100
        """).fetchall()
        conn.close()
        for r in relations:
            d = dict(r)
            rows_html += f"<tr><td>{d['source_name']}</td><td>{d['type']}</td><td>{d['target_name']}</td></tr>\n"
    except Exception:
        rows_html = "<tr><td colspan='3'>No relations found. Run 'codebookx analyze' first.</td></tr>"
    
    return f"""<h1>Codebase-X Knowledge Graph Viewer</h1>
<p>Database: {db_path}</p>
<p>API endpoints: /api/files, /api/symbols, /api/relations</p>
<p>🚀 <strong><a href="/graph">View Interactive Knowledge Graph</a></strong></p>
<h2>🔗 Relations</h2>
<table border="1"><tr><th>Source Symbol</th><th>Type</th><th>Target Symbol</th></tr>
{rows_html}</table>
<p>💡 <strong>CONTAINS</strong> means a file includes a symbol. 
<strong>CALLS</strong> and <strong>IMPORTS</strong> show cross-file dependencies.</p>"""

@app.route("/api/relations")
def get_relations():
    try:
        conn = get_db_connection()
        relations = conn.execute("SELECT * FROM relations LIMIT 500").fetchall()
        conn.close()
        return jsonify([dict(r) for r in relations])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/files")
def get_files():
    try:
        conn = get_db_connection()
        files = conn.execute("SELECT * FROM files").fetchall()
        conn.close()
        return jsonify([dict(f) for f in files])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/symbols")
def get_symbols():
    try:
        limit = request.args.get("limit", 100, type=int)
        limit = min(limit, 200)
        conn = get_db_connection()
        total = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        symbols = conn.execute("SELECT * FROM symbols LIMIT ?", (limit,)).fetchall()
        conn.close()
        return jsonify({"symbols": [dict(s) for s in symbols], "total": total})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_server(port=8050, db_path=".codebook_cache.db"):
    app.config['DB_PATH'] = Path(db_path).resolve()
    app.run(host="127.0.0.1", port=port)
