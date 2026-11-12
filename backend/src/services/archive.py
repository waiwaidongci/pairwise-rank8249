"""Deterministic bootstrap journal and archive integrity checks.

The journal is stored in the run ``result`` column while the worker is
resampling.  It is written with one transaction per completed repeat, so an
interrupted worker can resume from the next repeat.  Draws are chained with
SHA-256; every value needed to reproduce the chain is also part of the frozen
snapshot key.  Final results carry an ``archive`` block with the same anchor.
"""
import hashlib
import json
import math

import numpy as np

ALGORITHM_VERSION = 'centered-ridge-bt-empirical-binomial-v1'
ARCHIVE_VERSION = 'centered-ridge-bt-archive-v1'
GENESIS_DIGEST = '0' * 64


class ArchiveCorruptError(ValueError):
    """Raised when a stored journal or result archive is incomplete or damaged."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def snapshot_key(snapshot, ridge, repeats, seed):
    """Same canonical payload as the published result fingerprint."""
    payload = {
        'version': ALGORITHM_VERSION,
        'snapshot': snapshot,
        'ridge': ridge,
        'repeats': repeats,
        'seed': seed,
    }
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def chained_digest(previous, draw):
    return hashlib.sha256((str(previous) + canonical(draw)).encode()).hexdigest()


def draws_digest(draws):
    digest = GENESIS_DIGEST
    for draw in draws:
        digest = chained_digest(digest, draw)
    return digest


def concept_order(concepts):
    return [{'id': c['id'], 'code': c['code'], 'title': c['title']} for c in concepts]


def _fail(message):
    raise ArchiveCorruptError(message)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number_list(value, length, label, draw_number):
    if not isinstance(value, list) or len(value) != length:
        _fail(f'第{draw_number}次重复的{label}长度不符')
    for item in value:
        if not _is_number(item) or not math.isfinite(float(item)):
            _fail(f'第{draw_number}次重复的{label}含非有限数值')


def _validate_edges(stored_order, stored_totals, edges):
    if stored_order != edges:
        _fail('留档的比较边顺序或票数与冻结快照不符')
    if stored_totals != [edge['total'] for edge in edges]:
        _fail('留档的固定总票数与冻结快照不符')


def _validate_draw(draw, index, edges, vertices):
    number = index + 1
    if not isinstance(draw, dict):
        _fail(f'第{number}次重复不是留档对象')
    if not _is_int(draw.get('repeat')) or draw.get('repeat') != number:
        _fail(f'第{number}次重复的序号不连续或不是整数')
    wins = draw.get('wins')
    if not isinstance(wins, list) or len(wins) != len(edges):
        _fail(f'第{number}次重复的实际胜票边数不符')
    for edge_index, value in enumerate(wins):
        total = edges[edge_index]['total']
        if not _is_int(value) or value < 0 or value > total:
            _fail(f'第{number}次重复第{edge_index + 1}条边的胜票不在0–{total}范围内')
    _number_list(draw.get('abilities'), vertices, '能力向量', number)
    _number_list(draw.get('ranks'), vertices, '名次向量', number)
    _number_list(draw.get('first_share'), vertices, '首位分摊向量', number)
    gradient = draw.get('gradient_max')
    if not _is_number(gradient) or not math.isfinite(float(gradient)):
        _fail(f'第{number}次重复的投影梯度不是有限数值')
    if float(gradient) > 1e-6:
        _fail(f'第{number}次重复未通过一阶条件（梯度 {float(gradient):.4g}）')


def _validate_draws(draws, expected_count, edges, vertices):
    if not isinstance(draws, list) or len(draws) != expected_count:
        _fail('重复留档的记录数与已完成次数不一致')
    for index, draw in enumerate(draws):
        _validate_draw(draw, index, edges, vertices)


def _derived_rows(draws, concepts):
    """Recompute the published aggregates from archived draws."""
    rr = np.asarray([draw['ranks'] for draw in draws])
    ss = np.asarray([draw['first_share'] for draw in draws])
    rows = []
    for i, concept in enumerate(concepts):
        q = np.quantile(rr[:, i], [.025, .5, .975])
        frequency = {}
        for value in rr[:, i]:
            key = str(float(value))
            frequency[key] = frequency.get(key, 0) + 1
        frequency = dict(sorted(frequency.items(), key=lambda item: float(item[0])))
        rows.append({
            'id': concept['id'], 'code': concept['code'], 'title': concept['title'],
            'rank_low': float(q[0]), 'rank_median': float(q[1]), 'rank_high': float(q[2]),
            'first_share': float(ss[:, i].mean()), 'rank_frequency': frequency,
        })
    return rows


def validate_derived(result, concepts, draws):
    """Check that published aggregates match the archived draws."""
    rows = result.get('concepts')
    if not isinstance(rows, list) or len(rows) != len(concepts):
        _fail('结果的作品行数与冻结快照不符')
    expected = _derived_rows(draws, concepts)
    for row, ref in zip(rows, expected):
        if not isinstance(row, dict):
            _fail('结果的作品行结构损坏')
        for key, value in ref.items():
            if row.get(key) != value:
                _fail(f'结果中{ref["code"]}的{key}与留档重拟合不一致')
        for key in ('ability', 'rank', 'comparisons'):
            if key not in row or not _is_number(row.get(key)) or not math.isfinite(float(row[key])):
                _fail(f'结果中{ref["code"]}的{key}不是有限数值')
    probabilities = result.get('probabilities')
    if not isinstance(probabilities, list) or len(probabilities) != len(concepts):
        _fail('结果的胜率矩阵尺寸不符')
    for i, row in enumerate(probabilities):
        if not isinstance(row, list) or len(row) != len(concepts):
            _fail('结果的胜率矩阵尺寸不符')
        for j, value in enumerate(row):
            if i == j:
                if value is not None:
                    _fail('胜率矩阵对角应为空')
            elif not _is_number(value) or not 0. <= float(value) <= 1.:
                _fail('胜率矩阵含无效概率值')


def _validate_concept_order(stored, concepts):
    if stored != concept_order(concepts):
        _fail('留档的作品顺序与冻结快照不符')


def validate_journal(journal, expected_key, repeats, concepts, edges):
    """Validate and return (completed, draws, digest) for an in-progress journal."""
    if not isinstance(journal, dict):
        _fail('重复留档结构损坏')
    if journal.get('version') != ALGORITHM_VERSION:
        _fail('重复留档的算法版本与当前版本不一致')
    if journal.get('snapshot_key') != expected_key:
        _fail('重复留档与冻结票数快照不一致')
    if not _is_int(journal.get('repeats')) or journal.get('repeats') != repeats:
        _fail('重复留档的目标重复数与任务不一致')
    _validate_concept_order(journal.get('concept_order'), concepts)
    _validate_edges(journal.get('edge_order'), journal.get('edge_totals'), edges)
    draws = journal.get('draws')
    completed = journal.get('repeats_completed')
    if not _is_int(completed):
        _fail('重复留档缺少已完成的重复次数')
    if completed < 0 or completed > repeats:
        _fail('重复留档的已完成次数超出目标范围')
    _validate_draws(draws, completed, edges, len(concepts))
    digest = draws_digest(draws)
    if journal.get('digest') != digest:
        _fail('重复留档摘要不一致，已完成重复可能已损坏')
    return completed, draws, digest


def validate_completed(result, expected_key, repeats, concepts, edges):
    """Validate a completed result and its archive anchor; return draws digest."""
    if not isinstance(result, dict):
        _fail('结果留档不是对象')
    archive = result.get('archive')
    if not isinstance(archive, dict):
        _fail('结果缺少留档校验信息，不能视为完整结果')
    if archive.get('version') != ARCHIVE_VERSION:
        _fail('结果留档版本与当前校验规则不一致')
    if archive.get('snapshot_key') != expected_key:
        _fail('结果留档与冻结票数快照不一致')
    if not _is_int(archive.get('repeats_completed')) or archive.get('repeats_completed') != repeats:
        _fail('全部重复尚未齐全，不能视为完整结果')
    _validate_edges(result.get('edge_order'), result.get('edge_totals'), edges)
    _validate_draws(result.get('draws'), repeats, edges, len(concepts))
    digest = draws_digest(result['draws'])
    if archive.get('draws_digest') != digest:
        _fail('结果留档摘要不一致，重复记录可能已损坏')
    if result.get('resumed_from') != archive.get('resumed_from'):
        _fail('结果留档的恢复来源计数不一致')
    if not _is_int(result.get('resumed_from')) or result.get('resumed_from') < 0 or result.get('resumed_from') > repeats:
        _fail('结果留档缺少有效的恢复来源计数')
    validate_derived(result, concepts, result['draws'])
    return digest


def result_archive(key, completed, digest, resumed_from):
    return {
        'version': ARCHIVE_VERSION,
        'snapshot_key': key,
        'repeats_completed': completed,
        'draws_digest': digest,
        'resumed_from': resumed_from,
        'algorithm_version': ALGORITHM_VERSION,
    }


def journal_payload(key, repeats, concepts, edges, totals, draws, digest, resumed_from=0):
    return {
        'version': ALGORITHM_VERSION,
        'snapshot_key': key,
        'repeats': repeats,
        'concept_order': concept_order(concepts),
        'edge_order': edges,
        'edge_totals': list(totals),
        'draws': draws,
        'repeats_completed': len(draws),
        'digest': digest,
        'resumed_from': resumed_from,
    }


def _expected_layout(snapshot):
    from src.services.comparison_graph import prepare_graph
    try:
        concepts, edges, _ = prepare_graph(snapshot.get('concepts') or [], snapshot.get('pairs') or [])
    except Exception as exc:
        raise ArchiveCorruptError(f'冻结票数快照损坏：{exc}')
    return concepts, edges


def inspect_run(status, stored, snapshot, ridge, repeats, seed):
    """Inspect a run's stored archive without raising.

    Returns a compact progress report.  ``state`` is one of ``empty``,
    ``journal``, ``completed`` or ``corrupt``.  Only ``completed`` means every
    repeat is present and the archive chain verifies.  A journal left behind by
    an interrupted worker is resumable.
    """
    report = {
        'state': 'empty',
        'completed': 0,
        'repeats': repeats,
        'digest': GENESIS_DIGEST,
        'resumed_from': 0,
        'message': '',
        'snapshot_key': '',
        'resumable': False,
    }
    try:
        if status != 'completed' and (not isinstance(stored, dict) or not stored):
            return report
        concepts, edges = _expected_layout(snapshot)
        key = snapshot_key(snapshot, ridge, repeats, seed)
        report['snapshot_key'] = key
        if status == 'completed':
            digest = validate_completed(stored, key, repeats, concepts, edges)
            report.update(state='completed', completed=repeats, digest=digest,
                          resumed_from=int(stored.get('resumed_from') or 0))
            return report
        if status not in ('queued', 'running', 'failed', 'completed'):
            return report
        completed, draws, digest = validate_journal(stored, key, repeats, concepts, edges)
        report.update(state='journal', completed=completed, digest=digest,
                      resumed_from=int(stored.get('resumed_from') or 0),
                      resumable=completed < repeats)
        return report
    except ArchiveCorruptError as exc:
        report.update(state='corrupt', completed=0, message=str(exc))
        return report
