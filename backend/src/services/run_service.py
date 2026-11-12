"""Run lifecycle: freeze snapshot, execute with resumable bootstrap journal."""
import json
from pathlib import Path
from sqlalchemy import select,func
from fastapi import HTTPException
from redis import Redis
from src.config.settings import REDIS_URL
from src.models import RankingRun
from src.services.study_service import get_study
from src.services.concept_service import list_concepts
from src.services.pair_service import list_pairs
from src.services.rank_bootstrap import analyze_ranking
from src.services.archive import ArchiveCorruptError,inspect_run
from src.constants.enums import RunStatus
redis=Redis.from_url(REDIS_URL)
def list_runs(db):return db.scalars(select(RankingRun).order_by(RankingRun.id.desc())).all()
def get_run(db,id):
    item=db.get(RankingRun,id)
    if not item:raise HTTPException(404,'拟合任务不存在')
    return item

def create_run(db,data):
    study=get_study(db,data.study_id)
    if db.scalar(select(RankingRun).where(RankingRun.name==data.name)):raise HTTPException(409,'任务名称已存在')
    if db.scalar(select(func.count()).select_from(RankingRun).where(RankingRun.status.in_(['queued','running'])))>=3:raise HTTPException(429,'已有3项活动任务，请等待完成')
    concepts=[{k:getattr(c,k) for k in ('id','code','title','medium')} for c in list_concepts(db,study.id)]
    pairs=[{k:getattr(c,k) for k in ('id','left_id','right_id','left_wins','right_wins','source')} for c in list_pairs(db,study.id)]
    snapshot={'study':{'id':study.id,'code':study.code,'name':study.name,'source_file':study.source_file},'concepts':concepts,'pairs':pairs}
    # A new task always starts a fresh journal; no earlier archive is reused.
    item=RankingRun(**data.model_dump(),snapshot=snapshot,status=RunStatus.QUEUED,result={});db.add(item);db.commit()
    # Database is authoritative; worker scans queued tasks even if Redis notification fails.
    try:redis.lpush('pairwise:runs',str(item.id))
    except Exception:pass
    return item

def _resumable_journal(item):
    """Return the validated journal to resume, or None for a fresh start.

    Any non-empty stored payload that is not a verified in-progress journal is
    reported as a damaged archive; it is never silently discarded.
    """
    stored=item.result
    if not isinstance(stored,dict) or not stored:return None
    progress=inspect_run(item.status,stored,item.snapshot,item.ridge,item.repeats,item.seed)
    if progress['state']=='corrupt':raise ArchiveCorruptError(progress['message'])
    if progress['state']=='completed':raise ArchiveCorruptError('任务状态与完整留档不一致，需人工检查')
    return stored

def resume_run(db,id):
    """Requeue an interrupted task whose journal still verifies."""
    item=get_run(db,id)
    if item.status!=RunStatus.FAILED:raise HTTPException(409,'只有已中断的任务需要手动恢复')
    progress=inspect_run(item.status,item.result,item.snapshot,item.ridge,item.repeats,item.seed)
    if progress['state']=='corrupt':raise HTTPException(409,'重复留档损坏，不能恢复：'+progress['message'])
    if progress['state']!='journal' or not progress['resumable']:raise HTTPException(409,'没有可恢复的重复留档，请提交新任务')
    item.status=RunStatus.QUEUED;item.error='';db.commit()
    try:redis.lpush('pairwise:runs',str(item.id))
    except Exception:pass
    return item

def execute_run(db,id):
    item=get_run(db,id)
    if item.status not in ('queued','running'):return
    try:
        journal=_resumable_journal(item)
    except ArchiveCorruptError as exc:
        item.status=RunStatus.FAILED;item.error=('重复留档损坏，未重新计算：'+str(exc))[:600];db.commit();return
    item.status=RunStatus.RUNNING;db.commit()
    # Provenance: the first point at which an interrupted journal was resumed.
    # It is recorded, never used in the bootstrap arithmetic.
    resumed_from=int(journal.get('resumed_from') or 0) if journal else 0
    if journal and resumed_from<=0:resumed_from=int(journal.get('repeats_completed') or 0)
    def persist(next_journal):
        # One transaction per completed repeat: a crash loses at most the
        # repeat being computed, and the next worker resumes at this boundary.
        item.result=next_journal;db.commit()
    try:
        result=analyze_ranking(item.snapshot,item.ridge,item.repeats,item.seed,lambda:Path('/tmp/heartbeat').touch(),resume=journal,persist=persist,resumed_from=resumed_from)
        item.result=result;item.error='';item.status=RunStatus.COMPLETED;db.commit()
        # The database row is authoritative.  The artifact is a copy for
        # inspection; a read-only /artifacts volume must not fail a good study.
        try:
            Path('/artifacts').mkdir(exist_ok=True);Path(f'/artifacts/run-{id}.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))
        except OSError as exc:
            item.error=('结果已写入数据库，但留档文件写入失败：'+str(exc))[:600];db.commit()
    except ArchiveCorruptError as exc:
        db.rollback();item=get_run(db,id);item.status=RunStatus.FAILED;item.error=('重复留档损坏，未重新计算：'+str(exc))[:600];db.commit()
    except Exception as exc:
        db.rollback();item=get_run(db,id);item.status=RunStatus.FAILED;item.error=str(exc)[:600];db.commit()
