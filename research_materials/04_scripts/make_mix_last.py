import json
from pathlib import Path

ROOT = Path(r"C:\lora_exp")

NOVEL = ROOT / "01_data" / "shadow_family.jsonl"
S2    = ROOT / "01_data" / "train_s2.fixed.jsonl"

OUT_TAIL = ROOT / "01_data" / "train_shadow_tail_s2.jsonl"
OUT_INT  = ROOT / "01_data" / "train_shadow_interleave_s2_10.jsonl"

NOVEL_BLOCK = 10  # 10화마다 S2 1회

def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                raise ValueError(f"[{p.name}] line {ln} JSON error: {e}")
    return rows

def write_jsonl(p: Path, rows):
    with p.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    if not NOVEL.exists():
        raise FileNotFoundError(f"NOVEL not found: {NOVEL}")
    if not S2.exists():
        raise FileNotFoundError(f"S2 not found: {S2}")

    novel = load_jsonl(NOVEL)
    s2 = load_jsonl(S2)
    
    if len(s2) == 0:
        raise ValueError("S2 jsonl is empty.")
    s2_line = s2[-1]
    
    # (A) Tail: 소설 + S2 1회(맨 뒤)
    tail = novel + [s2_line]
    write_jsonl(OUT_TAIL, tail)
    print("WROTE TAIL:", OUT_TAIL, "lines=", len(tail))
    
    # (B) Interleave: 소설 10화마다 S2 1회
    inter = []
    i = 0
    while i < len(novel):
        inter.extend(novel[i:i+NOVEL_BLOCK])
        i += NOVEL_BLOCK
        inter.append(s2_line)
    write_jsonl(OUT_INT, inter)
    print("WROTE INTERLEAVE:", OUT_INT, "lines=", len(inter))

if __name__ == "__main__":
    main()