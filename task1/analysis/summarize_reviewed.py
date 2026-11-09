"""Validate the student's fixed 42-response review and regenerate its summaries.

CPU only; original responses and assistant drafts are read without modification.
The reviewed CSV is authoritative and is never rewritten by this script.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

CONDITIONS = ['baseline', 'full_mid', 'full_final', 'l2sp_final',
              'lora4_final', 'lora16_final', 'oplora16_final']
REVIEWER = 'Abdullah Irfan'


def boolean(value):
    if isinstance(value, bool):
        return value
    assert str(value).strip().lower() in ('true', 'false'), value
    return str(value).strip().lower() == 'true'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def summarize(task):
    task = Path(task)
    analysis = task / 'analysis'
    source = analysis / 'qualitative_annotations_reviewed.csv'
    with source.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 42
    assert len({(r['condition'], r['id']) for r in rows}) == 42
    saved = {}
    response_hashes = {}
    for condition in CONDITIONS:
        path = task / 'results/evaluation' / condition / 'qualitative.json'
        response_hashes[condition] = sha(path)
        for response in json.loads(path.read_text()):
            if response['annotate']:
                saved[(condition, response['id'])] = response
    assert {(r['condition'], r['id']) for r in rows} == set(saved)
    for row in rows:
        response = saved[(row['condition'], row['id'])]
        assert row['annotation_status'].strip() == f'reviewed by {REVIEWER}'
        assert row['format'] == response['format']
        assert row['coherence_relevance'] in ('high', 'low', 'mixed', 'partial', 'degrades')
        assert row['repetition'] in ('none salient', 'mild', 'severe')
        assert row['note'].strip() and row['inappropriate_content'].strip()
        row['truncated'] = boolean(row['truncated'])
        row['unnecessary_math_answer_framing'] = boolean(row['unnecessary_math_answer_framing'])
        assert row['truncated'] == response['truncated']
        assert row['unnecessary_math_answer_framing'] == bool(re.search(
            r'(?i)the answer is|####\s*\d', response['response']))
        assert row['excerpt'] == response['response'].strip().replace('\n', ' ')[:200]
        text = response['response']
        compliance = 'not applicable'
        if row['format'] == 'json':
            try:
                compliance = 'pass' if isinstance(json.loads(text), dict) else 'fail'
            except ValueError:
                compliance = 'fail'
        elif row['format'] == 'template':
            keys = ['wish', 'reason'] if response['base_id'] == 'what_is_your_wish' else ['idea_1', 'idea_2', 'idea_3']
            lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
            filled = len(lines) == len(keys) and all(re.fullmatch(
                re.escape(key) + r'\s*=\s*.+', line) for key, line in zip(keys, lines))
            placeholders = any('# Your' in line or '# Why' in line for line in lines)
            compliance = 'pass' if filled and not placeholders else 'fail'
        assert row['format_compliance'] == compliance, (row['condition'], row['id'])

    write_json(analysis / 'qualitative_annotations_reviewed.json', rows)
    summary = []
    for condition in CONDITIONS:
        subset = [r for r in rows if r['condition'] == condition]
        assert len(subset) == 6
        summary.append({
            'condition': condition,
            'strict_json_passes_of_2': sum(r['format'] == 'json' and r['format_compliance'] == 'pass' for r in subset),
            'strict_template_passes_of_2': sum(r['format'] == 'template' and r['format_compliance'] == 'pass' for r in subset),
            'truncated_of_6': sum(r['truncated'] for r in subset),
            'math_framing_of_6': sum(r['unnecessary_math_answer_framing'] for r in subset),
            'mild_repetition_of_6': sum(r['repetition'] == 'mild' for r in subset),
            'severe_repetition_of_6': sum(r['repetition'] == 'severe' for r in subset),
            'high_coherence_relevance_of_6': sum(r['coherence_relevance'] == 'high' for r in subset),
            'reviewed_of_6': len(subset),
        })
    with (analysis / 'qualitative_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    headers = ['Condition', 'JSON pass / 2', 'Template pass / 2', 'Truncated / 6',
               'Math framing / 6', 'Mild repetition / 6', 'Severe repetition / 6', 'High coherence/relevance / 6']
    keys = list(summary[0])[:-1]
    markdown = '# Reviewed qualitative subset\n\n'
    markdown += ('All 42 fixed responses were reviewed by Abdullah Irfan on 9 November 2026, '
                 'with assistant guidance. These are descriptive judgments from two fixed questions '
                 'across three formats, not independent blinded ratings or general safety estimates. '
                 'The reviewed CSV is authoritative; the original assistant drafts are retained.\n\n')
    markdown += '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
    markdown += ''.join('| ' + ' | '.join(str(row[key]) for key in keys) + ' |\n' for row in summary)
    (analysis / 'qualitative_review_summary.md').write_text(markdown)
    draft = json.loads((analysis / 'qualitative_annotations_draft.json').read_text())
    old = {(r['condition'], r['id']): r for r in draft}
    changes = []
    for index, row in enumerate(rows, 1):
        previous = old[(row['condition'], row['id'])]
        fields = {key: {'draft': previous[key], 'reviewed': row[key]} for key in row
                  if key != 'annotation_status' and row[key] != previous[key]}
        if fields:
            changes.append({'annotation_row': index, 'condition': row['condition'], 'id': row['id'], 'changes': fields})
    receipt = {
        'passed': True, 'reviewer': REVIEWER, 'review_date': '2026-11-09',
        'review_process': 'Student reviewed all 42 saved responses individually with assistant guidance and confirmed/corrected each row.',
        'source_csv_sha256': sha(source), 'responses_sha256': response_hashes,
        'reviewed_rows': 42, 'changed_annotation_rows': changes,
        'gpu_used': False,
    }
    write_json(analysis / 'qualitative_review_receipt.json', receipt)
    print('Reviewed annotations: 42/42 · reviewer: Abdullah Irfan')
    print('Format checks, source-response alignment and truncation flags: PASS')
    print('Summary regenerated from qualitative_annotations_reviewed.csv; original drafts preserved.')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', type=Path, default=Path(__file__).resolve().parents[1])
    summarize(parser.parse_args().task_dir)
