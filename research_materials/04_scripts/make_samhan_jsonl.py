import json
from pathlib import Path

ROOT = Path(r"C:\lora_exp")
SRC_DIR = ROOT / "01_data" / "samhan_empire"

OUT_NOVEL = ROOT / "01_data" / "train_samhan_01_14.jsonl"
OUT_MIX_10TO1 = ROOT / "01_data" / "train_samhan_01_14_s2_10to1.jsonl"

S2_FILE = ROOT / "01_data" / "train_s2.fixed.jsonl"  # 기존 S2 (fixed) 경로
NOVEL_BLOCK = 10  # 소설 10개 샘플마다 S2 1개 삽입

# 파일 정렬 순서 강제: prologue -> ch01..ch14
def ordered_md_files(src_dir: Path):
    files = []
    pro = src_dir / "prologue.md"
    if pro.exists():
        files.append(pro)
    else:
        pro = src_dir / "prologue"
        if pro.exists():
            files.append(pro)

    # ch01~ch14 (확장자 .md 또는 확장자 없음 둘 다 허용)
    for i in range(1, 15):
        name = f"ch{i:02d}"
        p1 = src_dir / f"{name}.md"
        p2 = src_dir / name
        if p1.exists():
            files.append(p1)
        elif p2.exists():
            files.append(p2)
        else:
            raise FileNotFoundError(f"Missing chapter file: {name}(.md)")
    return files

def read_text(path: Path) -> str:
    # 윈도우에서 BOM이 섞이는 경우가 있어 utf-8-sig 우선
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8")

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    if not rows:
        raise ValueError(f"S2 jsonl is empty: {path}")
    return rows

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    # 1) 소설(프롤로그~14화) -> 챕터 단위 샘플로 JSONL 생성
    md_files = ordered_md_files(SRC_DIR)

    novel_rows = []
    for p in md_files:
        title = p.stem  # prologue, ch01 ...
        body = read_text(p).strip()
        # 샘플 앞에 구분 헤더를 붙여두면 later 분석/디버깅에 유리
        text = f"# {title}\n\n{body}\n"
        novel_rows.append({"text": text})
    
    write_jsonl(OUT_NOVEL, novel_rows)
    print("WROTE:", OUT_NOVEL, "lines=", len(novel_rows))
    
    # 2) 소설 + S2 (10:1 interleave)
    s2_rows = load_jsonl(S2_FILE)
    s2_line = s2_rows[-1]  # S2가 짧으니 1줄을 반복 삽입(기존 실험과 동일)
    
    mixed = []
    i = 0
    while i < len(novel_rows):
        mixed.extend(novel_rows[i:i+NOVEL_BLOCK])
        i += NOVEL_BLOCK
        mixed.append(s2_line)
    
    write_jsonl(OUT_MIX_10TO1, mixed)
    print("WROTE:", OUT_MIX_10TO1, "lines=", len(mixed))

if __name__ == "__main__":
    main()