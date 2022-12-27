
from dataclasses import dataclass, field, fields
import datetime
import os, sys, pickle, glob, gc
import json
import pandas as pd
from typing import List
import numpy as np
from dataclasses_json import dataclass_json
from tqdm.notebook import tqdm

try:
    import cudf
except:
    print('can not import cudf')


@dataclass_json
@dataclass
class Config:
    # 対象のevent typeを持つlist (click: 0, cart: 1, order: 2)
    # 何も指定しない場合は、全type選択
    target_types: List[int] = field(default_factory=lambda: [0, 1, 2])

    # eventに重みをつける関数。WeightFuncMixinで登録されている名前を利用
    weight_func: str = 'type_weight_1_1_1'

    # 1sessionでevent数の最小しきい値。この値未満のデータは弾く
    min_event_threshold: int = 30

    # 共起とみなす最大の時間(単位: sec)。この値を超えて発生したevent同士は、同一sessionでも
    max_sec_threshold: int = 24 * 60 * 60

    # 1aidにつき, 保存する件数。これ以下のデータは切り捨てる
    save_topk: int = 15

    # 使用ファイル(ファイルのpathを書いておく)
    source_data: List[str] = field(default_factory=lambda: [])

    def is_same_setting(self, config):
        """同じ設定のcovis matrixかどうかをチェック"""
        my_conf_fields = [conf_field.name for conf_field in fields(self)]
        comparison_fields = [conf_field.name for conf_field in fields(config)]

        if my_conf_fields != comparison_fields:
            return False

        for field in my_conf_fields:
            if self.__getattribute__(field) != config.__getattribute__(field):
                return False

        return True

    def save(self, output_dir: str):
        with open(output_dir, 'w') as f:
            f.write(self.to_json())

    @staticmethod
    def from_json(json_data: dict):
        return Config(**json_data)


class WeightFuncMixin:
    def type_weight_1_1_1(self, df):
        type_weight = {0: 1, 1: 1, 2: 1}
        return df.type_y.map(type_weight)

    def type_weight_1_6_3(self, df):
        type_weight = {0: 1, 1: 6, 2: 3}
        return df.type_y.map(type_weight)

    def time_weight_v1(self, df):
        return 1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800)


class CovisMatrixGenerator:

    DISK_PIECES = 4
    READ_CT = 5
    SIZE = 1.86e6 / DISK_PIECES

    @staticmethod
    def get_same_setting_covismatrix(config, covis_dir):
        files = glob.glob(covis_dir + '/*/*')

        for file_ in files:
            if file_.endswith('config.json'):
                with open(file_, 'r') as f:
                    comparison = Config.from_json(f.read())
            if config.is_same_setting(comparison):
                return '/'.join(file_.split('/')[:-1])

        return

    def __init__(self, weight_func_mixin: WeightFuncMixin, use_gpu=False):

        self.weight_func_mixin = weight_func_mixin
        self.use_gpu = use_gpu

    def _set_weight_func(self, func_name):
        return getattr(self.weight_func_mixin, func_name)

    def _set_chunk(self, length):
        self.CHUNK = int(np.ceil(length / 6))

    type_labels = {'clicks':0, 'carts':1, 'orders':2}
    def read(self, f):
        df = pd.read_parquet(f)
        df.ts = (df.ts/1000).astype('int32')
        df['type'] = df['type'].map(self.type_labels).astype('int8')
        if self.use_gpu:
            return cudf.DataFrame(df)
        return df

    def generate(
        self, config: Config,
            output_dir=f"{(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d-%H%M%S')}"):

        os.makedirs(output_dir)

        files = config.source_data
        self._set_chunk(len(files))
        weight_func = self._set_weight_func(config.weight_func)
        # COMPUTE IN PARTS FOR MEMORY MANGEMENT
        for PART in tqdm(range(self.DISK_PIECES), desc='CUNK'):

            # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
            # => OUTER CHUNKS
            for j in tqdm(range(6), desc='OUTER CHUNK'):
                a = j*self.CHUNK
                b = min( (j+1)*self.CHUNK, len(files) )

                # => INNER self.CHUNKS
                for k in tqdm(range(a,b,self.READ_CT), desc='INNER CUNK'):
                    # READ FILE
                    df = [self.read(files[k])]
                    for i in range(1,self.READ_CT):
                        if k+i<b: df.append(self.read(files[k+i]))
                    if self.use_gpu:
                        df = cudf.concat(df,ignore_index=True,axis=0)
                    else:
                        df = pd.concat(df, ignore_index=True, axis=0)
                    df = df.loc[df['type'].isin(config.target_types)]
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
                    df = df[['session', 'aid_x', 'aid_y','ts_x', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
                    df['wgt'] = weight_func(df)
                    df = df[['aid_x','aid_y','wgt']]
                    df.wgt = df.wgt.astype('float32')
                    df = df.groupby(['aid_x','aid_y']).wgt.sum()
                    # COMBINE INNER self.CHUNKS
                    if k==a: tmp2 = df
                    else: tmp2 = tmp2.add(df, fill_value=0)
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
            tmp.to_pandas().to_parquet(f'{output_dir}/part{PART}.pqt')

        config.save(output_dir + '/config.json')
