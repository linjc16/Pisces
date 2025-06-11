def safe_round(value, ndigits=4):
    if isinstance(value, float):
        return round(value, ndigits)
    return value

