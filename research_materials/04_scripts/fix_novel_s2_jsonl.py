import json, pathlib

src = pathlib.Path(r"01_data\train_novel_s2.jsonl")
dst = pathlib.Path(r"01_data\train_novel_s2.fixed.jsonl")

raw = src.read_text(encoding="utf-8")
obj = {"text": raw}
dst.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")
print("WROTE:", dst)