"""Pure shadow diagnostic step; checkpoints belong to CausalObjectSurface records.

No geometry substitution, world publication or independent runtime state owner.
"""
from copy import deepcopy
import json
import numpy as np


def advance_checkpoint(checkpoint,index,row,limits, *, preserve_missing_reference=False):
    """Pure diagnostic step; checkpoint must come from the previous step.

    This is an offline validation primitive, not a separate runtime owner.
    A missing checkpoint is legal only at the beginning of the source sequence.
    """
    if type(preserve_missing_reference) is not bool:raise ValueError('invalid gap policy')
    for key in ('speed_m_s','acceleration_m_s2','maximum_anchor_age_s'):
        if not np.isfinite(limits[key]) or limits[key]<=0:raise ValueError('invalid limit')
    if checkpoint is None:
        if index!=0:raise ValueError('missing prior diagnostic checkpoint')
        state=dict(last_row=-1,last_timestamp=None,anchor=None,velocity=None,interval=None,identity=None,limits=dict(limits))
    else:
        state=deepcopy(checkpoint)
    if checkpoint is not None and state.get('preserve_missing_reference',False)!=preserve_missing_reference:
        raise ValueError('checkpoint gap policy mismatch')
    if preserve_missing_reference:
        state['preserve_missing_reference']=True
    if state['last_row']!=index-1 or state['limits']!=limits:raise ValueError('checkpoint sequence or policy mismatch')
    source=row['inputs']['chest'];timestamp=int(source['depth_ns'])
    if state['last_timestamp'] is not None and timestamp<=state['last_timestamp']:raise ValueError('nonmonotonic timestamp')
    item=dict(row=index,status='input_unavailable',blockers=[])
    features=[f for f in row.get('object_local_task_features',[]) if f['feature_id']=='pour_outlet' and f['status']=='available_current']
    if len(features)>1:raise ValueError('ambiguous feature')
    if not features:
        if not preserve_missing_reference:
            state.update(anchor=None,velocity=None,interval=None)
    else:
        feature=features[0]
        identity=json.dumps([feature['gauge_id'],feature['stream_key']],sort_keys=True)
        if state['identity'] is not None and state['identity']!=identity:raise ValueError('checkpoint identity changed')
        state['identity']=identity
        if int(feature['as_of_depth_timestamp_ns'])!=timestamp or feature['input_evidence_hash']!=source['input_evidence_hash']:
            raise ValueError('feature source mismatch')
        pose=np.asarray(row['table_from_initial_object'],float)
        point=(pose@np.r_[feature['feature_geometry']['point_m'],1])[:3]
        if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('invalid point')
        if state['anchor'] is None:
            item['status']='initialization_unchecked';state['anchor']=[index,timestamp,point.tolist()]
        else:
            old_index,old_timestamp,old_point=state['anchor'];dt=(timestamp-old_timestamp)/1e9
            item.update(reference_row=old_index,dt_s=dt)
            if dt>limits['maximum_anchor_age_s']:item['blockers'].append('reference_expired')
            velocity=(point-np.asarray(old_point))/dt
            item['speed_m_s']=float(np.linalg.norm(velocity))
            if item['speed_m_s']>limits['speed_m_s']:item['blockers'].append('speed_exceeded')
            if state['velocity'] is not None:
                item['acceleration_m_s2']=float(np.linalg.norm(velocity-np.asarray(state['velocity']))/((dt+state['interval'])/2))
                if item['acceleration_m_s2']>limits['acceleration_m_s2']:item['blockers'].append('acceleration_exceeded')
            if item['blockers']:item['status']='rejected_shadow'
            else:
                item['status']='admitted_shadow' if state['velocity'] is not None else 'velocity_only_unchecked_acceleration'
                state.update(anchor=[index,timestamp,point.tolist()],velocity=velocity.tolist(),interval=dt)
    state.update(last_row=index,last_timestamp=timestamp)
    return item,state

