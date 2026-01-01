import json, pathlib

NOVEL = pathlib.Path("01_data/shadow_family.jsonl")
S2    = pathlib.Path("01_data/train_s2.fixed.jsonl")
OUT   = pathlib.Path("01_data/train_shadow_tail_s2.jsonl")

def load(p):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

novel_rows = load(NOVEL)
s2_rows = load(S2)

# 소설 전체 + S2 마지막 1줄
mixed = novel_rows + [s2_rows[-1]]

with OUT.open("w", encoding="utf-8") as f:
    for r in mixed:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("DONE:", OUT, "lines =", len(mixed))