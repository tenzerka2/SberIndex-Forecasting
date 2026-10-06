"""Append manually transcribed search results to data/news/corpus.jsonl.

Input on stdin: one record per line, fields separated by ' || ':
  query_id || rank || url || published (ISO or NA) || title || snippet (verbatim excerpt only: no analyst annotations)
Collected with the Exa web-search tool (results returned to the analyst, not to the container,
because the container's egress policy blocks news sites); retrieved_at is the collection date.
"""
import json, sys
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "data" / "news" / "corpus.jsonl"
seen = set()
if OUT.exists():
    seen = {(json.loads(l)["query_id"], json.loads(l)["url"]) for l in OUT.read_text().splitlines() if l.strip()}
n = 0
with open(OUT, "a") as f:
    for line in sys.stdin:
        if not line.strip():
            continue
        q, r, url, pub, title, snip = [x.strip() for x in line.split(" || ")]
        if (q, url) in seen:
            continue
        f.write(json.dumps({"query_id": q, "rank": int(r), "url": url, "published": None if pub == "NA" else pub,
                            "title": title, "snippet": snip, "collector": "exa_web_search",
                            "retrieved_at": "2026-10-06"}, ensure_ascii=False) + "\n")
        n += 1
print("appended", n)
