"""Validate source syntax and review-copy hygiene without importing or fitting models."""
import argparse
import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote


def _local_asset(name):
    return name.startswith(('data/raw/', 'artifacts/'))


def _fallback_ignored_local_files(root, names, reason):
    """Recognize only the repository's explicit local-directory ignore policy.

    Without Git, tracked files cannot be distinguished from untracked files.
    An unknown negation therefore disables this fallback rather than hiding a
    potentially publishable file. All other directories remain fully checked.
    """
    ignore_file = root / '.gitignore'
    rules = []
    if ignore_file.is_file():
        rules = [line.strip() for line in ignore_file.read_text(encoding='utf-8-sig').splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
    known_exceptions = set()
    unknown_negations = any(rule.startswith('!') and rule not in known_exceptions for rule in rules)
    roots = []
    if not unknown_negations:
        for prefix, rule in [('data/raw/', '/data/raw/*'), ('artifacts/', '/artifacts/**')]:
            if rule in rules:
                roots.append(prefix)
    ignored = {name for name in names if any(name.startswith(prefix) for prefix in roots)}
    return ignored, {'method': 'restricted-directory-fallback', 'reason': reason,
                     'recognized_ignore_roots': roots,
                     'unknown_negation_disables_fallback': unknown_negations,
                     'limit': 'No Git index is available; only exact local-directory rules in .gitignore are recognized.'}, []


def _ignored_local_files(root, names):
    """Ask Git about local inputs/outputs; never exclude other ignored assets."""
    if not names:
        return set(), {'method': 'not-needed'}, []
    try:
        result = subprocess.run(['git', '-C', str(root), 'check-ignore', '-z', '--stdin'],
                                input='\0'.join(names) + '\0', text=True, encoding='utf-8',
                                capture_output=True, timeout=10, check=False)
    except FileNotFoundError:
        return _fallback_ignored_local_files(root, names, 'git-unavailable')
    except (OSError, subprocess.TimeoutExpired):
        return set(), {'method': 'git-check-ignore-error'}, ['Could not verify local-file Git ignore status.']
    if result.returncode in (0, 1):
        # Without --no-index, tracked files are intentionally never ignored.
        ignored = set(result.stdout.split('\0')) - {''}
        if not ignored.issubset(names):
            return set(), {'method': 'git-check-ignore-error'}, ['Unexpected Git ignore response.']
        return ignored, {'method': 'git-check-ignore', 'tracked_files_excluded': False}, []
    if 'not a git repository' in result.stderr.lower():
        return _fallback_ignored_local_files(root, names, 'no-git-repository')
    return set(), {'method': 'git-check-ignore-error'}, ['Could not verify local-file Git ignore status.']


def validate(root):
    root = Path(root).resolve()
    issues = []
    count = {'files': 0, 'python_sources_compiled': 0, 'json_files_parsed': 0,
             'markdown_links_checked': 0, 'total_bytes': 0, 'ignored_local_files': 0}
    skip = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist'}
    binary_suffixes = {'.cbm', '.pt', '.pth', '.joblib', '.pkl', '.pickle', '.parquet',
                       '.npz', '.npy', '.onnx', '.sqlite', '.db'}
    machine_path = re.compile(r'\b[A-Za-z]:[\\/](?:Users|Program Files|Windows)[\\/]', re.I)
    required = ['README.md', 'pyproject.toml', '.gitignore',
                'NOTICE.md', 'LICENSE.md',
                'docs/evaluation.md']
    for name in required:
        if not (root / name).is_file():
            issues.append(f'Missing required review file: {name}')
    files = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(p in skip or p.startswith('.venv') for p in rel.parts) or not path.is_file():
            continue
        files.append(path)
    local_names = [path.relative_to(root).as_posix() for path in files
                   if _local_asset(path.relative_to(root).as_posix())]
    ignored, ignore_policy, ignore_issues = _ignored_local_files(root, local_names)
    issues.extend(ignore_issues)
    count['ignored_local_files'] = len(ignored)
    for path in files:
        rel = path.relative_to(root)
        name = rel.as_posix()
        if name in ignored:
            continue
        count['files'] += 1
        count['total_bytes'] += path.stat().st_size
        if path.suffix.lower() in binary_suffixes or path.suffix.lower() in {'.log', '.pid', '.jsonl'}:
            issues.append(f'Generated/runtime asset in review copy: {name}')
        if _local_asset(name):
            issues.append(f'Local input/output content in review copy: {name}')
            # Reject the boundary violation without opening private input/output content.
            continue
        if path.stat().st_size > 8_000_000:
            issues.append(f'Unexpected large file: {name}')
        if path.suffix.lower() == '.png':
            continue
        try:
            text = path.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:
            issues.append(f'Unexpected non-text asset: {name}')
            continue
        if machine_path.search(text):
            issues.append(f'Machine-specific absolute path: {name}')
        if path.suffix == '.py':
            try:
                compile(text, name, 'exec')
                count['python_sources_compiled'] += 1
            except SyntaxError as exc:
                issues.append(f'Python syntax: {name}:{exc.lineno}: {exc.msg}')
        elif path.suffix == '.json':
            try:
                def reject_constant(value):
                    raise ValueError('non-finite JSON number')
                json.loads(text, parse_constant=reject_constant)
                count['json_files_parsed'] += 1
            except (ValueError, TypeError) as exc:
                issues.append(f'Invalid strict JSON: {name}: {type(exc).__name__}')
        elif path.suffix == '.toml':
            try:
                tomllib.loads(text)
            except ValueError:
                issues.append(f'Invalid TOML: {name}')
        elif path.suffix == '.md':
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                link = link.strip().split(' "')[0].strip('<>')
                if not link or link.startswith(('#', 'http://', 'https://', 'mailto:')):
                    continue
                target = unquote(link.split('#', 1)[0])
                count['markdown_links_checked'] += 1
                if not (path.parent / target).resolve().exists():
                    issues.append(f'Broken local documentation link: {name} -> {target}')
    return {'status': 'pass' if not issues else 'fail',
            'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'checks': count, 'issues': issues,
            'publication_scope': {'local_file_ignore_policy': ignore_policy,
                                  'ignored_local_files': sorted(ignored)},
            'models_fitted': 0, 'model_predictions_generated': 0,
            'model_imports_executed': 0,
            'limits': ['Syntax and file-boundary checks do not establish numerical reproduction.',
                       'Source permissions and data provenance need maintainer review.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', type=Path)
    args = parser.parse_args()
    report = validate(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    raise SystemExit(0 if report['status'] == 'pass' else 1)


if __name__ == '__main__':
    main()
