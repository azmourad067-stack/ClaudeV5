import json
import pandas as pd
def load(path, cols, key='csv'):
    s=json.load(open(path))['result']
    i=s.find('\n[{'); j=s.rfind('}]')
    rows=json.loads(s[i+1:j+2])
    txt="\n".join(r[key] for r in rows if r.get(key))
    recs=[l.split('|') for l in txt.split('\n') if l]
    return pd.DataFrame(recs, columns=cols)
