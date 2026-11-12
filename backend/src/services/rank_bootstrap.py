"""Conditional count bootstrap: fixed n_ij, empirical p_ij, not a posterior.

The random stream advances once per repeat, so a journal of completed repeats
can be replayed into the same NumPy generator state.  Resuming therefore
continues at the next repeat and produces the same draws and final result as an
uninterrupted run.
"""
from collections import Counter
import numpy as np
from scipy.special import expit
from src.services.comparison_graph import prepare_graph
from src.services.bt_solver import fit_bt,average_ranks
from src.services.archive import (
    GENESIS_DIGEST,
    ArchiveCorruptError,
    chained_digest,
    concept_order,
    journal_payload,
    result_archive,
    snapshot_key,
    validate_journal,
)

def analyze_ranking(snapshot,ridge,repeats,seed,heartbeat=None,resume=None,persist=None,resumed_from=0):
    """Fit observed counts and optionally resume a journal of completed repeats.

    ``resume`` is a journal produced by a previous partial run of this task.
    ``persist(journal)`` is called after every newly completed repeat so the
    worker can commit one repeat atomically.  ``resumed_from`` records the
    journal length inherited from the interrupted attempt (0 for fresh runs);
    it is provenance only and does not affect the numbers.
    """
    if not isinstance(repeats,int) or not 50<=repeats<=200:raise ValueError('bootstrap重复须为50–200')
    concepts,edges,graph=prepare_graph(snapshot['concepts'],snapshot['pairs']);n=len(concepts)
    key=snapshot_key(snapshot,ridge,repeats,seed)
    order=concept_order(concepts)
    totals=[e['total'] for e in edges];observed=np.array([e['wins']/e['total'] for e in edges]);totals_array=np.array(totals)
    rng=np.random.default_rng(seed);draws=[];digest=GENESIS_DIGEST;start=0
    if resume is not None:
        start,draws,digest=validate_journal(resume,key,repeats,concepts,edges)
        draws=list(draws)  # never mutate the caller's journal
        # Replaying consumes the same binomial draws without refitting.  If the
        # journal's wins differ, the archived repeats are not from this seed and
        # the archive is reported as damaged instead of silently restarted.
        for index in range(start):
            if rng.binomial(totals_array,observed).tolist()!=draws[index]['wins']:
                raise ArchiveCorruptError(f'第{index+1}次重复的实际胜票与固定种子不一致')
            if heartbeat:heartbeat()
    for b in range(start,repeats):
        wins=rng.binomial(totals_array,observed)
        resampled=[{**e,'wins':int(w)} for e,w in zip(edges,wins)]
        values,diag=fit_bt(n,resampled,ridge);r,s=average_ranks(values)
        draw={'repeat':b+1,'wins':wins.tolist(),'abilities':values.tolist(),'ranks':r.tolist(),'first_share':s.tolist(),'gradient_max':diag['gradient_max']}
        draws.append(draw);digest=chained_digest(digest,draw)
        if persist:persist(journal_payload(key,repeats,concepts,edges,totals,draws,digest,resumed_from))
        if heartbeat:heartbeat()
    theta,diagnostic=fit_bt(n,edges,ridge);rank,_=average_ranks(theta)
    rr=np.asarray([d['ranks'] for d in draws]);ss=np.asarray([d['first_share'] for d in draws])
    rows=[]
    for i,c in enumerate(concepts):
        q=np.quantile(rr[:,i],[.025,.5,.975]);freq=Counter(str(float(x)) for x in rr[:,i])
        rows.append({**c,'ability':float(theta[i]),'rank':float(rank[i]),'rank_low':float(q[0]),'rank_median':float(q[1]),'rank_high':float(q[2]),'first_share':float(ss[:,i].mean()),'rank_frequency':dict(sorted(freq.items(),key=lambda x:float(x[0]))),'comparisons':graph['degree'][i]})
    matrix=[[None if i==j else float(expit(theta[i]-theta[j])) for j in range(n)] for i in range(n)]
    return {'concepts':rows,'probabilities':matrix,'graph':graph,'diagnostic':diagnostic,'ridge':ridge,'repeats':repeats,'seed':seed,'draws':draws,'edge_totals':totals,'edge_order':edges,'fingerprint':key,'archive':result_archive(key,repeats,digest,resumed_from),'resumed_from':resumed_from,'snapshot':snapshot,'tie_rule':'average rank for abilities within 1e-8; first share split equally among tied first','interval':'2.5%/97.5% empirical rank quantiles, linearly interpolated; not posterior or simultaneous confidence limits','resampling':'independent Binomial(n_ij, observed wins_ij/n_ij), fixed pair totals'}
