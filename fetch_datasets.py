"""从 OpenML 取一批真实的小规模表格分类数据集，缓存到本地。
选取范围对齐这类模型的适用区间：几百到几千行、特征不多、类别数少。"""

import os
import pickle
import warnings

import numpy as np
import openml

warnings.filterwarnings("ignore")
CACHE = "datasets.pkl"

# OpenML-CC18 里常用于表格基础模型评测的一批小数据集
TASKS = {
    "credit-g": 31, "diabetes": 37, "tic-tac-toe": 50, "vehicle": 54,
    "kc1": 1067, "pc1": 1068, "banknote": 1462, "blood-transfusion": 1464,
    "climate-crashes": 1467, "wdbc": 1510, "steel-plates": 1504,
    "phoneme": 1489, "qsar-biodeg": 1494, "wilt": 40983, "churn": 40701,
    "ilpd": 1480, "spambase": 44, "sonar": 40, "ionosphere": 59, "vowel": 307,
}


def main():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        print(f"已有缓存：{len(d)} 个数据集")
        return d

    out = {}
    for name, did in TASKS.items():
        try:
            ds = openml.datasets.get_dataset(did, download_data=True,
                                             download_qualities=False,
                                             download_features_meta_data=True)
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
            X = X.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c)
            X = X.astype(float)
            X = X.fillna(X.median())
            y = np.asarray(y)
            classes, y = np.unique(y, return_inverse=True)
            Xv = X.values
            keep = np.isfinite(Xv).all(0) & (Xv.std(0) > 0)
            Xv = Xv[:, keep]
            if Xv.shape[1] < 3 or len(classes) > 10 or len(y) < 300:
                print(f"  跳过 {name}（形状不合适）")
                continue
            out[name] = (Xv.astype(np.float64), y.astype(int))
            print(f"  {name:<20} {Xv.shape[0]:>5} 行 x {Xv.shape[1]:>3} 列，{len(classes)} 类")
        except Exception as e:
            print(f"  {name} 取失败：{type(e).__name__} {str(e)[:70]}")

    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    print(f"\n共取得 {len(out)} 个数据集，已缓存")
    return out


if __name__ == "__main__":
    main()
