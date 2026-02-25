import argparse, itertools, json, io, gzip

def open_auto(path):
    with open(path, 'rb') as f:
        sig = f.read(2)
    if sig == b'\x1f\x8b':  # gzip 魔数
        return gzip.open(path, 'rt', encoding='utf-8', errors='ignore')
    # 二进制读取后用 TextIOWrapper 容错解码
    return io.TextIOWrapper(open(path, 'rb'), encoding='utf-8', errors='ignore')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--validate', action='store_true')
    args = ap.parse_args()

    with open_auto(args.input) as fin, open(args.output, 'w', encoding='utf-8') as fout:
        for line in itertools.islice(fin, args.n):
            if args.validate:
                try:
                    json.loads(line)
                except Exception:
                    continue
            if not line.endswith('\n'):
                line += '\n'
            fout.write(line)

if __name__ == '__main__':
    main()