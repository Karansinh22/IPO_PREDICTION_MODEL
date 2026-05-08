from scraper import _fetch_sub_api, _parse_sub_rows, _fetch_gmp_api, _parse_gmp_rows

sub_rows = _fetch_sub_api()
sub_dict = _parse_sub_rows(sub_rows)
print('SUB KEYS (first 15):')
for k in list(sub_dict.keys())[:15]:
    print(f'  [{k}] -> QIB={sub_dict[k]["qib"]} RII={sub_dict[k]["rii"]}')

print()
gmp_rows = _fetch_gmp_api()
gmp_dict = _parse_gmp_rows(gmp_rows)
print('GMP KEYS (first 15):')
for k in list(gmp_dict.keys())[:15]:
    print(f'  [{k}]')

print()
print('OVERLAP CHECK:')
for k in list(sub_dict.keys()):
    if k in gmp_dict:
        print(f'  MATCH: [{k}]')
    else:
        print(f'  MISS:  [{k}]')
