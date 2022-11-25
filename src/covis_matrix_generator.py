
from dataclasses import dataclass
import datetime

import os, sys, pickle, glob, gc

# GPU使えるときは, cudfを読み込む
try:
    import cudf as pd
except:
    import pandas as pd

from utils import read_file


@dataclass
class Config:
    # eventのtypeに重み(click: 0, cart: 1, order: 2)
    type_weight: dict

    # 1sessionでevent数の最小しきい値。この値未満のデータは弾く
    min_event_threshold: int

    # 共起とみなす最大の時間(単位: sec)。この値を超えて発生したevent同士は、同一sessionでも
    max_sec_threshold: int

    # 1aidにつき, 保存する件数。これ以下のデータは切り捨てる
    save_topk: int

    # 保存先
    output_dir: str = f"covis_matrix/{(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d-%H%M%S')}"



class CovisMatrixGenerator:

    DISK_PIECES = 4
    READ_CT = 5
    CHUNK = 6
    SIZE = 1.86e6 / DISK_PIECES

    def generate(self, config, files):
        # COMPUTE IN PARTS FOR MEMORY MANGEMENT
        for PART in range(self.DISK_PIECES):
            print('### DISK PART',PART+1)

            # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
            # => OUTER CHUNKS
            for j in range(6):
                a = j*self.CHUNK
                b = min( (j+1)*self.CHUNK, len(files) )
                print(f'Processing files {a} thru {b-1} in groups of {self.READ_CT}...')

                # => INNER self.CHUNKS
                for k in range(a,b,self.READ_CT):
                    # READ FILE
                    df = [read_file(files[k])]
                    for i in range(1,self.READ_CT):
                        if k+i<b: df.append( read_file(files[k+i]) )
                    df = pd.concat(df,ignore_index=True,axis=0)
                    df = df.sort_values(['session','ts'],ascending=[True,False])
                    # USE TAIL OF SESSION
                    df = df.reset_index(drop=True)
                    df['n'] = df.groupby('session').cumcount()
                    df = df.loc[df.n < config.min_event_threshold].drop('n',axis=1)
                    # CREATE PAIRS
                    df = df.merge(df,on='session')
                    df = df.loc[ ((df.ts_x - df.ts_y).abs()< config.max_sec_threshold) & (df.aid_x != df.aid_y) ]
                    # MEMORY MANAGEMENT COMPUTE IN PARTS
                    df = df.loc[(df.aid_x >= PART*self.SIZE)&(df.aid_x < (PART+1)*self.SIZE)]
                    # ASSIGN WEIGHTS
                    df = df[['session', 'aid_x', 'aid_y','type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
                    df['wgt'] = df.type_y.map(config.type_weight)
                    df = df[['aid_x','aid_y','wgt']]
                    df.wgt = df.wgt.astype('float32')
                    df = df.groupby(['aid_x','aid_y']).wgt.sum()
                    # COMBINE INNER self.CHUNKS
                    if k==a: tmp2 = df
                    else: tmp2 = tmp2.add(df, fill_value=0)
                    print(k,', ',end='')
                print()
                # COMBINE OUTER self.CHUNKS
                if a==0: tmp = tmp2
                else: tmp = tmp.add(tmp2, fill_value=0)
                del tmp2, df
                gc.collect()
            # CONVERT MATRIX TO DICTIONARY
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
            tmp = tmp.reset_index(drop=True)
            tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
            tmp = tmp.loc[tmp.n<config.save_topk].drop('n',axis=1)
            # SAVE PART TO DISK (convert to pandas first uses less memory)
            os.makedirs(config.output_dir, exist_ok=True)
            tmp.to_pandas().to_parquet(f'{config.output_dir}/part_{PART}.pqt')
