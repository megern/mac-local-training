"""Atomic local review bundles. Hashes check file consistency, not authenticity."""
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import tempfile

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write_bundle(output, report, title):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Review bundle must use a fresh directory')
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if len(payload.encode()) > 64 * 1024 * 1024:
        raise ValueError('Report exceeds 64 MiB bundle limit')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix='.review-pending-', dir=output.parent))
    try:
        (pending / 'report.json').write_text(payload)
        esc = lambda value: html.escape(str(value), quote=True)
        sections = []
        for key, value in report.items():
            if isinstance(value, list):
                content = ''.join(('<details><summary>Record ' + str(i + 1) + '</summary><pre>' + esc(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></details>' for i, row in enumerate(value[:1000])))
                if len(value) > 1000:
                    content += '<p>Showing first 1000 records; JSON contains all ' + str(len(value)) + '.</p>'
            else:
                content = '<pre>' + esc(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>'
            sections.append('<section><h2>' + esc(key.replace('_', ' ')) + '</h2>' + content + '</section>')
        document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>' + esc(title) + '</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#f5f7fa;color:#152437}section{background:white;padding:18px;margin:16px 0;border:1px solid #d4deea;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;padding:10px}a{color:#185ba2}</style><h1>' + esc(title) + '</h1><p>Local review report · <a href="report.json">Complete JSON evidence</a> · no external scripts or services</p>' + ''.join(sections) + '</html>'
        (pending / 'report.html').write_text(document)
        manifest = {'schema_version': 1, 'title': title, 'files_sha256': {name: digest(pending / name) for name in ('report.json', 'report.html')}, 'scope': 'Local file consistency only; not independent authentication or certification.'}
        (pending / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists():
            raise ValueError('Destination appeared while building report')
        os.rename(pending, output)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return manifest

def audit_bundle(output):
    output = Path(output).resolve()
    manifest_path = output / 'manifest.json'
    if manifest_path.stat().st_size > 65536:
        raise ValueError('Oversized manifest')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or set(manifest.get('files_sha256', {})) != {'report.json', 'report.html'}:
        raise ValueError('Unsupported review-bundle manifest')
    for name, expected in manifest['files_sha256'].items():
        path = output / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError('Changed review artifact: ' + name)
    return {'status': 'consistent', 'files_verified': 2, 'scope': manifest['scope']}
