"""Attach versioned SberIndex names/regions/OKTMO to the verified identity mapping."""
from pathlib import Path
import argparse,hashlib,json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sbx.municipal_graph import territory_metadata


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dictionary',type=Path,default=ROOT/'data/raw/hackathon/t_dict_municipal_districts.xlsx')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/graph_warning')
    args=ap.parse_args()
    m=pd.read_csv(args.output/'identity_map.csv.gz')
    d=pd.read_excel(args.dictionary,dtype={'oktmo':str})
    audit={'source_url':'https://www.sberbank.com/common/files/t_dict_municipal.rar',
        'source_page':'https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities',
        'received_on':'2026-10-07','dictionary_rows':len(d),
        'dictionary_sha256':hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
        'version_rule':'year_from <= year < year_to','years':{},
        'notes':['Metadata is descriptive and is not used by graph-warning prediction models.',
                 'Recovering ID from full-history values is an identity audit, not a predictive feature.',
                 'Source download used a per-request TLS-verification exception after the bank certificate chain could not be verified; no trust settings were changed.',
                 'Retrieved public URLs via the download script of IvanFrolovskiy/sberindex-atlas; no model code or experiment results copied.']}
    for year in [2023,2024]:
        joined=territory_metadata(m,d,year)
        joined.to_csv(args.output/f'territory_map_{year}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        mismatch=joined.loc[joined.mo.ne(joined.municipal_district_name),['territory_id','mo','municipal_district_name','region_name']]
        audit['years'][str(year)]={'matched':len(joined),'regions':joined.region_name.nunique(),
            'name_differences':mismatch.to_dict('records')}
    (args.output/'metadata_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print({year:{'matched':v['matched'],'regions':v['regions'],'name_differences':len(v['name_differences'])} for year,v in audit['years'].items()})


if __name__=='__main__':main()
