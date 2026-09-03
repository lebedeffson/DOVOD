from pathlib import Path

root = Path(__file__).resolve().parents[1]
forbidden = {'.docx', '.pdf', '.dotm', '.pptx'}
bad = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in forbidden]
if bad:
    raise SystemExit('Publication/editorial files found:\n' + '\n'.join(map(str, bad)))
print('Public repository hygiene: PASS')
