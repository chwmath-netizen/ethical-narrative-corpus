import json, pathlib

src = pathlib.Path(r"01_data\train_s2.jsonl")
dst = pathlib.Path(r"01_data\train_s2.fixed.jsonl")

# 파일 전체를 그대로 읽어서(개행 포함) JSON 문자열로 안전하게 이스케이프
raw = src.read_text(encoding="utf-8")

# 이미 {"text": "..."} 형태로 만들려고 하던 원본이라면,
# 중복을 피하려고 앞뒤를 좀 정리할 수 있지만, 일단은 '그대로' 하나의 텍스트로 학습해도 됩니다.
obj = {"text": raw}

dst.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")
print("WROTE:", dst)