"""
scraper.py — Live IPO data from investorgain.com's internal JSON API.
Fetches real-time IPO GMP, subscription, price, and status data.
Cached in memory for 15 minutes to avoid hammering the source.
"""

import requests
import re
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- In-memory cache (15 min TTL) ----
_cache = {'data': None, 'ts': 0}
CACHE_TTL = 900  # seconds

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/122.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://www.investorgain.com/',
    'Origin': 'https://www.investorgain.com',
    'Accept': 'application/json, text/plain, */*',
}


def _strip_html(text):
    """Remove HTML tags from a string."""
    if not text:
        return ''
    return re.sub(r'<[^>]+>', '', str(text)).strip()


def _parse_gmp(gmp_html):
    """Extract numeric GMP value from the HTML GMP field."""
    if not gmp_html:
        return 0
    # Look for a number like 3.5 or -- inside <b> tag
    m = re.search(r'<b>\s*(-?\d+(?:\.\d+)?)\s*<\/b>', gmp_html)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0


def _parse_price(price_str):
    """Extract a float from a price string like '175' or '₹175'."""
    if not price_str:
        return 0
    clean = re.sub(r'[^\d.]', '', _strip_html(price_str))
    try:
        return float(clean) if clean else 0
    except ValueError:
        return 0


def _parse_subscription(sub_str):
    """Parse subscription like '0.25x' → 0.25, or '-' → 0."""
    if not sub_str or sub_str == '-':
        return 0
    clean = re.sub(r'[xX,\s]', '', _strip_html(sub_str))
    try:
        return float(clean)
    except ValueError:
        return 0


def _parse_size(size_str):
    """Parse IPO size like '150.06 ' → 150.06."""
    if not size_str:
        return 0
    clean = re.sub(r'[^\d.]', '', _strip_html(size_str))
    try:
        return float(clean) if clean else 0
    except ValueError:
        return 0


def _get_status(name_html, open_date, close_date):
    """Determine IPO status from badge class in Name HTML and date range."""
    if not name_html:
        return 'Unknown'
    if 'bg-success' in name_html or '>O<' in name_html:
        return 'Open'
    if 'bg-warning' in name_html or '>U<' in name_html:
        return 'Upcoming'
    if 'bg-danger' in name_html or '>C<' in name_html:
        return 'Closed'
    if '>L<' in name_html:
        return 'Listed'
    # Fallback: determine from dates
    try:
        today = datetime.utcnow().date()
        open_d = datetime.strptime(open_date, '%Y-%m-%d').date() if open_date else None
        close_d = datetime.strptime(close_date, '%Y-%m-%d').date() if close_date else None
        if open_d and close_d:
            if today < open_d:
                return 'Upcoming'
            elif open_d <= today <= close_d:
                return 'Open'
            else:
                return 'Closed'
    except Exception:
        pass
    return 'Closed'


def _fetch_live_ipos():
    """Fetch live IPO list from investorgain.com webnodejs API."""
    # Build current year params
    now = datetime.utcnow()
    year = now.year
    fy = f'{year}-{str(year + 1)[2:]}'  # e.g. 2026-27

    url = (
        f'https://webnodejs.investorgain.com/cloud/new/report/'
        f'data-read/331/1/4/{year}/{fy}/0/all?search=&v=12-49'
    )

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json().get('reportTableData', [])
    except Exception as e:
        logger.error(f'investorgain API error: {e}')
        return []


def _parse_rows(rows):
    """Convert raw API rows into clean IPO dicts."""
    result = []
    for row in rows:
        try:
            name = row.get('~ipo_name', '')
            if not name:
                continue

            # Remove trailing " IPO" / " SME IPO" suffix for cleaner display
            clean_name = re.sub(r'\s+(IPO|SME IPO|InvIT)$', '', name, flags=re.IGNORECASE).strip()

            offer_price = _parse_price(row.get('Price (₹)', ''))
            gmp = _parse_gmp(row.get('GMP', ''))
            sub = _parse_subscription(row.get('Sub', ''))
            size = _parse_size(row.get('IPO Size (₹ in cr)', ''))
            gmp_pct = float(row.get('~gmp_percent_calc') or 0)
            status = _get_status(
                row.get('Name', ''),
                row.get('~Srt_Open', ''),
                row.get('~Srt_Close', '')
            )
            open_date = row.get('Open', '')
            close_date = row.get('Close', '')
            listing_date = row.get('Listing', '')
            category = row.get('~IPO_Category', 'IPO')  # 'IPO' or 'SME'

            result.append({
                'name': clean_name,
                'full_name': name,
                'status': status,
                'offer_price': offer_price,
                'gmp': gmp,
                'gmp_pct': gmp_pct,
                'total': sub,         # subscription x
                'qib': 0,             # not available in this API
                'hni': 0,
                'rii': 0,
                'size': size,
                'open_date': open_date,
                'close_date': close_date,
                'listing_date': listing_date,
                'category': 'SME' if category == 'SME' else 'Mainboard',
                'actual_gain': None,  # only known post-listing via separate source
            })
        except Exception as e:
            logger.warning(f'Row parse error: {e}')
            continue

    return result


def get_indian_market_ipos():
    """
    Returns a list of current & recent Indian IPOs with live data.
    Results are cached for 15 min to reduce load on the source API.
    """
    global _cache
    now = time.time()

    if _cache['data'] is not None and (now - _cache['ts']) < CACHE_TTL:
        logger.info('Returning cached IPO data')
        return _cache['data']

    logger.info('Fetching fresh IPO data from investorgain.com API...')
    rows = _fetch_live_ipos()
    ipos = _parse_rows(rows)

    if not ipos:
        logger.warning('Live fetch returned no IPOs, using fallback stub.')
        # Return a minimal stub so the app doesn't break
        ipos = [
            {
                'name': 'Data Loading...',
                'full_name': 'Fetching live IPO data failed. Try again shortly.',
                'status': 'Closed',
                'offer_price': 0, 'gmp': 0, 'gmp_pct': 0,
                'total': 0, 'qib': 0, 'hni': 0, 'rii': 0,
                'size': 0, 'open_date': '', 'close_date': '',
                'listing_date': '', 'category': 'Mainboard', 'actual_gain': None,
            }
        ]

    _cache['data'] = ipos
    _cache['ts'] = now
    logger.info(f'Fetched {len(ipos)} live IPOs.')
    return ipos


if __name__ == '__main__':
    import json
    data = get_indian_market_ipos()
    print(f'Total IPOs: {len(data)}')
    for ipo in data[:10]:
        print(f"  [{ipo['status']:8}] {ipo['name']:40} Price=₹{ipo['offer_price']} GMP=₹{ipo['gmp']} ({ipo['gmp_pct']}%) Sub={ipo['total']}x  [{ipo['category']}]")
