"""Review multiple completed research exports locally."""
import argparse
import json
from pathlib import Path
from .study_evidence import audit_export
from .report_bundle import write_bundle

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('studies', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        if not 1 <= len(a.studies) <= 32:
            raise ValueError('Supply 1..32 completed study exports')
        results = []
        for path in a.studies:
            try:
                results.append({'status': 'consistent', 'evidence': audit_export(path)})
            except (ValueError, OSError, KeyError, TypeError) as error:
                results.append({'status': 'rejected', 'study': path.name, 'error': str(error)})
        report = {'tool': 'local-research-catalog', 'studies': results, 'accepted_studies': sum((row['status'] == 'consistent' for row in results)), 'scope': 'Includes negative acceptance gates and rejected artifacts; consistency is not novel research or independent authentication.'}
        write_bundle(a.output, report, 'Local research evidence catalog')
        print(json.dumps({'studies': len(results), 'accepted': report['accepted_studies']}))
        return 0 if report['accepted_studies'] == len(results) else 2
    except (ValueError, OSError) as error:
        p.exit(2, str(error) + '\n')
if __name__ == '__main__':
    raise SystemExit(main())
