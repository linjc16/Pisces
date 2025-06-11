_store = {}

def log_scalar(name, value, weight=1.0, priority=0, round=4):
    _store[name] = {
        "value": value,
        "weight": weight,
        "priority": priority,
    }

def reset():
    _store.clear()

def log_start_time(name, priority=0, round=0):
    _store[f"{name}_start"] = True

def aggregate(name):
    return _store.get(name, {})

def get_smoothed_value(name):
    return _store.get(name, {}).get("value", 0)

