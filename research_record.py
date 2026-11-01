"""Append research milestones with artifact hashes; no inference or scheduling."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LOG=ROOT/'RESEARCH_EVENTS.jsonl'

def record(event,artifacts,outcome,limitations):
    now=datetime.datetime.now(datetime.timezone.utc)
    previous=LOG.read_text(encoding='utf8').splitlines()[-1] if LOG.exists() and LOG.stat().st_size else None
    hashes={}
    for name in artifacts:
        p=ROOT/name; assert p.is_file(),name
        hashes[Path(name).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    entry=dict(timestamp_utc=now.isoformat(),timestamp_ist=now.astimezone(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),event=event,outcome=outcome,limitations=limitations,artifact_sha256=hashes,previous_event_sha256=hashlib.sha256(previous.encode()).hexdigest() if previous else None)
    with LOG.open('a',encoding='utf8') as f: f.write(json.dumps(entry,ensure_ascii=False,sort_keys=True)+'\n')
    return entry
