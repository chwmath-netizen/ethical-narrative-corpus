import json, sys

p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        s = line.strip()
        if not s:
            continue
        try:
            json.loads(s)
        except Exception as e:
            print("BAD LINE", i)
            print(s[:200])
            print("ERR", e)
            sys.exit(1)

print("ALL OK")