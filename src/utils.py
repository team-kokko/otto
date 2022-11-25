try:
    import cudf as pd
except:
    import pandas as pd

def read_file(f):
    return pd.DataFrame( data_cache[f] )

type_labels = {'clicks':0, 'carts':1, 'orders':2}
def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts/1000).astype('int32')
    df['type'] = df['type'].map(type_labels).astype('int8')
    return df
