# baselines/threshold_autoscaler.py
def threshold_autoscaler(obs, current_active, max_servers, up_thresh=0.7, down_thresh=0.3):
    """
    obs: observation array with first max_servers entries as server utils
    current_active: current active server count
    returns: new_active (0..max_servers)
    """
    utils = obs[:max_servers]
    avg_util = sum(utils[:current_active]) / max(1, current_active)
    if avg_util > up_thresh and current_active < max_servers:
        return min(max_servers, current_active + 1)
    if avg_util < down_thresh and current_active > 1:
        return max(1, current_active - 1)
    return current_active
