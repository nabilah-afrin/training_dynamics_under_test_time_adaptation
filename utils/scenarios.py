ALL_SCENARIOS = ['clean_base', 'noisy_base', 'clean_tta', 'noisy_tta']

def resolve_scenarios(cfg):
    s = getattr(cfg, 'scenarios', 'all')
    if s in (None, 'all'):
        return list(ALL_SCENARIOS)
    if isinstance(s, str):
        s = [s]
    bad = [x for x in s if x not in ALL_SCENARIOS]
    if bad:
        raise ValueError(f"Unknown scenarios {bad}. Options: {ALL_SCENARIOS}")
    return s