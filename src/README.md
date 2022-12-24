## src について

OTTO コンペで、全体共通・よく使うコードを component 化している

### 事前準備

このコードを kaggle notebook で利用するには 2 つの事前準備が必要

-   github アクセス用の personal_access_token を作る。詳しくは、(こちら)[https://rfs.jp/server/git/github/personal_access_tokens.html]
-   kaggle の Add-ons/secret 機能を使って、↑ で作った token を保存する(key=personal_access_token)

### 利用方法

事前準備を行った上で、下記のコードを kaggle notebook に入れる

```
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
personal_access_token = user_secrets.get_secret("personal_access_token")

!rm -rf /kaggle/working/kaggle_otto
!git clone -b gcs https://$personal_access_token@github.com/coffeemountain/kaggle_otto.git

import sys
sys.path.append('/kaggle/working/kaggle_otto/src')

# 好きなcomponentをimportする
!pip install dataclasses_json
from covis_matrix_generator import *
```

### tips

-   secret は notebook 毎に on にする必要がある
-   merge 前の branch を取ってくるときは、下記のようにする
    `git clone -b <brand_name> https://$personal_access_token@github.com/coffeemountain/kaggle_otto.git`

### component 一覧

| コンポーネント名       | 概要                                 | サンプルコード                                                                                                |
| ---------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| covis_matrix_generator | 様々な設定で共訪問 matrix 自動生成機 | https://github.com/coffeemountain/kaggle_otto/blob/main/notebooks/candidate-rerank-with-my-covis-matrix.ipynb |
