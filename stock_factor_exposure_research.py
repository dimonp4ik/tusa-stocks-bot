"""Research mapping of ETF direction to shared daily benchmark direction."""
SOURCE='https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs'


def semiconductor_proxy_key(row):
    multiplier={'SOXLUSDT':1,'SOXSUSDT':-1}.get(row['symbol'])
    if multiplier is None:
        return None
    if row['direction'] not in ('LONG','SHORT'):
        raise ValueError('invalid trade direction')
    side=1 if row['direction']=='LONG' else -1
    return ('semiconductor_daily_proxy',side*multiplier)
