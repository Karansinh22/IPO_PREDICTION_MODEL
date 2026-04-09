"""
scraper.py — Live + Historical Indian IPO data.
Sources:
  1. investorgain.com GMP API  → Live/recent IPOs (status, GMP, price, dates)
  2. investorgain.com Sub API  → Live subscription with QIB / RII / HNI breakdown
  3. Internal Excel dataset    → 5-year historical archive with full fundamentals
Results are merged and cached for 15 minutes.
"""

import requests
import re
import logging
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_cache = {'data': None, 'ts': 0}
CACHE_TTL = 900   # 15 minutes

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://www.investorgain.com/',
    'Origin': 'https://www.investorgain.com',
    'Accept': 'application/json, text/plain, */*',
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_html(text):
    if not text:
        return ''
    return re.sub(r'<[^>]+>', '', str(text)).strip()


def _num(text, default=0):
    """Extract first float from any string."""
    clean = re.sub(r'[^\d.\-]', '', _strip_html(str(text)))
    # remove trailing dots
    clean = clean.rstrip('.')
    try:
        return float(clean) if clean and clean != '-' else default
    except ValueError:
        return default


def _parse_gmp(gmp_html):
    m = re.search(r'<b>\s*(-?\d+(?:\.\d+)?)\s*<\/b>', str(gmp_html))
    return float(m.group(1)) if m else 0


def _parse_sub_total(total_html):
    """Pull the first bold number out of a subscription HTML cell."""
    m = re.search(r'<b>\s*(-?\d+(?:\.\d+)?)\s*<\/b>', str(total_html))
    return float(m.group(1)) if m else _num(total_html)


def _status_from_html(name_html, open_date='', close_date=''):
    nh = str(name_html)
    if 'bg-success' in nh or '>O<' in nh:
        return 'Open'
    if 'bg-warning' in nh or '>U<' in nh:
        return 'Upcoming'
    if ('bg-primary' in nh or 'bg-danger' in nh or '>C<' in nh):
        return 'Closed'
    if '>L<' in nh:
        return 'Listed'
    # Fallback by date
    try:
        today = datetime.utcnow().date()
        od = datetime.strptime(open_date, '%Y-%m-%d').date() if open_date else None
        cd = datetime.strptime(close_date, '%Y-%m-%d').date() if close_date else None
        if od and cd:
            if today < od:   return 'Upcoming'
            if od <= today <= cd: return 'Open'
            return 'Closed'
    except Exception:
        pass
    return 'Closed'


def _clean_name(raw_name):
    """Remove common suffixes like ' IPO', ' SME IPO', ' BSE SME' for display."""
    name = re.sub(r'\s+(IPO|SME IPO|BSE SME|NSE SME|InvIT)\s*$', '', raw_name, flags=re.I)
    return name.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — investorgain GMP API (live/recent IPOs, last ~2 years)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_gmp_api():
    year = datetime.utcnow().year
    fy   = f'{year}-{str(year + 1)[2:]}'
    url  = (f'https://webnodejs.investorgain.com/cloud/new/report/'
            f'data-read/331/1/4/{year}/{fy}/0/all?search=&v=12-49')
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get('reportTableData', [])
    except Exception as e:
        logger.error(f'GMP API error: {e}')
        return []


def _parse_gmp_rows(rows):
    result = {}
    for row in rows:
        name = str(row.get('~ipo_name', '')).strip()
        if not name:
            continue
        result[name] = {
            'full_name':    name,
            'name':         _clean_name(name),
            'status':       _status_from_html(row.get('Name', ''),
                                              row.get('~Srt_Open', ''),
                                              row.get('~Srt_Close', '')),
            'offer_price':  _num(row.get('Price (₹)', 0)),
            'gmp':          _parse_gmp(row.get('GMP', '')),
            'gmp_pct':      _num(row.get('~gmp_percent_calc', 0)),
            'total':        _num(row.get('Sub', 0)),
            'qib':          0,
            'hni':          0,
            'rii':          0,
            'size':         _num(row.get('IPO Size (₹ in cr)', 0)),
            'open_date':    row.get('Open', ''),
            'close_date':   row.get('Close', ''),
            'listing_date': row.get('Listing', ''),
            'category':     'SME' if row.get('~IPO_Category') == 'SME' else 'Mainboard',
            'actual_gain':  None,
            'year':         None,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — investorgain Subscription API (live QIB / RII breakdown)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_sub_api():
    year = datetime.utcnow().year
    fy   = f'{year}-{str(year + 1)[2:]}'
    url  = (f'https://webnodejs.investorgain.com/cloud/new/report/'
            f'data-read/333/1/4/{year}/{fy}/0/all?search=&v=12-49')
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get('reportTableData', [])
    except Exception as e:
        logger.error(f'Sub API error: {e}')
        return []


def _parse_sub_rows(rows):
    """Returns dict keyed by full IPO name → {qib, hni, rii, total}"""
    result = {}
    for row in rows:
        # Name HTML contains the title attribute which is the clean name
        name_html = str(row.get('Name', ''))
        m = re.search(r'title="([^"]+)"', name_html)
        name = m.group(1).strip() if m else _strip_html(name_html).split('\n')[0].strip()
        if not name:
            continue
        result[name] = {
            'qib':   _num(row.get('QIB', 0)),
            'hni':   _num(row.get('NII', 0)),   # NII = Non-Institutional (HNI)
            'rii':   _num(row.get('RII', 0)),
            'total': _parse_sub_total(row.get('Total', '')),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — Internal Excel dataset (561 IPOs, 2010-2025)
# ─────────────────────────────────────────────────────────────────────────────

_EXCEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'dataset', 'raw_dataset', 'Initial Public Offering.xlsx')


def _load_excel_ipos(years=5):
    """Return historical IPOs from the last `years` years as a list of dicts."""
    try:
        df = pd.read_excel(_EXCEL_PATH)
        df.columns = [c.strip() for c in df.columns]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        cutoff = datetime.now() - timedelta(days=years * 365)
        df = df[df['Date'] >= cutoff].copy()
        df = df.dropna(subset=['IPO_Name'])
        df = df.sort_values('Date', ascending=False)

        result = []
        for _, row in df.iterrows():
            name = str(row.get('IPO_Name', '')).strip()
            if not name:
                continue
            listing_gain = row.get('Listing Gain')
            gain = float(listing_gain) if pd.notnull(listing_gain) else None
            dt = row['Date']
            result.append({
                'full_name':    name,
                'name':         name,
                'status':       'Listed',
                'offer_price':  _num(row.get('Offer Price', 0)),
                'gmp':          0,
                'gmp_pct':      0,
                'total':        _num(row.get('Total', 0)),
                'qib':          _num(row.get('QIB', 0)),
                'hni':          _num(row.get('HNI', 0)),
                'rii':          _num(row.get('RII', 0)),
                'size':         _num(row.get('Issue_Size(crores)', 0)),
                'open_date':    '',
                'close_date':   '',
                'listing_date': dt.strftime('%d-%b') if pd.notnull(dt) else '',
                'category':     'Mainboard',
                'actual_gain':  gain,
                'year':         int(dt.year) if pd.notnull(dt) else None,
            })
        logger.info(f'Loaded {len(result)} historical IPOs from Excel (last {years} years)')
        return result
    except Exception as e:
        logger.error(f'Excel load error: {e}')
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MERGE — Live (GMP + Sub) on top of Historical (Excel)
# ─────────────────────────────────────────────────────────────────────────────

def _build_combined():
    # Fetch all three sources in parallel-ish
    gmp_rows  = _fetch_gmp_api()
    sub_rows  = _fetch_sub_api()

    gmp_dict  = _parse_gmp_rows(gmp_rows)   # key = full IPO name
    sub_dict  = _parse_sub_rows(sub_rows)   # key = full IPO name

    # Merge subscription data into GMP dict
    for name, sdata in sub_dict.items():
        if name in gmp_dict:
            gmp_dict[name].update({
                'qib':   sdata['qib'],
                'hni':   sdata['hni'],
                'rii':   sdata['rii'],
                'total': sdata['total'] if sdata['total'] > 0 else gmp_dict[name]['total'],
            })
        else:
            # Subscription API may have IPOs not in GMP API (very recent closed)
            ipo = {
                'full_name': name,
                'name': _clean_name(name),
                'status': 'Closed',
                'offer_price': 0, 'gmp': 0, 'gmp_pct': 0,
                'total': sdata['total'], 'qib': sdata['qib'],
                'hni': sdata['hni'], 'rii': sdata['rii'],
                'size': 0, 'open_date': '', 'close_date': '',
                'listing_date': '', 'category': 'Mainboard',
                'actual_gain': None, 'year': None,
            }
            gmp_dict[name] = ipo

    live_ipos = list(gmp_dict.values())

    # Historical from Excel — skip any name already in live data
    live_names_lower = {i['name'].lower() for i in live_ipos}
    live_names_lower |= {i['full_name'].lower() for i in live_ipos}

    hist_ipos = _load_excel_ipos(years=5)
    deduped_hist = []
    for h in hist_ipos:
        if h['name'].lower() not in live_names_lower:
            deduped_hist.append(h)

    # Sort: Open first, then Upcoming, Closed, Listed (historical)
    order = {'Open': 0, 'Upcoming': 1, 'Closed': 2, 'Listed': 3}
    combined = sorted(live_ipos, key=lambda x: order.get(x['status'], 4)) + deduped_hist

    logger.info(f'Combined: {len(live_ipos)} live + {len(deduped_hist)} historical = {len(combined)} total')
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_indian_market_ipos():
    """Returns live + 5-year historical Indian IPOs. Cached for 15 min."""
    global _cache
    now = time.time()

    if _cache['data'] is not None and (now - _cache['ts']) < CACHE_TTL:
        logger.info(f'Returning {len(_cache["data"])} cached IPOs')
        return _cache['data']

    data = _build_combined()

    if not data:
        logger.warning('No IPO data fetched — returning empty list')
        data = []

    _cache['data'] = data
    _cache['ts'] = now
    return data


if __name__ == '__main__':
    ipos = get_indian_market_ipos()
    print(f'\nTotal IPOs: {len(ipos)}')
    print(f"Live:       {sum(1 for i in ipos if i['status'] in ('Open','Upcoming','Closed'))}")
    print(f"Historical: {sum(1 for i in ipos if i['status'] == 'Listed')}")
    print()
    for ipo in ipos[:15]:
        gain_str = f"Gain={ipo['actual_gain']}%" if ipo['actual_gain'] is not None else ''
        print(f"[{ipo['status']:8}] {ipo['name'][:40]:40} "
              f"P={ipo['offer_price']:>6} QIB={ipo['qib']:>5} RII={ipo['rii']:>5} "
              f"Sub={ipo['total']:>5}x  {gain_str}")
