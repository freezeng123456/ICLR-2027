"""把 PFN 的 setting 用具体数据打出来看：一个任务长什么样、模型实际看到什么、
先验（造表的程序）是什么、以及先验错配长什么样。纯 numpy，无需 GPU。"""

import numpy as np

RNG = np.random.default_rng(0)


def fmt_table(X, y, n_query=0, label_name="y"):
    """把特征矩阵和标签排成对齐的表格；最后 n_query 行的标签显示为 ?"""
    d = X.shape[1]
    head = "  行 |" + "".join(f"   x{j + 1:<5}" for j in range(d)) + f" |  {label_name}"
    lines = [head, "  " + "-" * (len(head) - 2)]
    n_ctx = len(X) - n_query
    for i in range(len(X)):
        cells = "".join(f" {X[i, j]:7.2f}" for j in range(d))
        tag = "?" if i >= n_ctx else f"{y[i]:.0f}" if float(y[i]).is_integer() else f"{y[i]:.2f}"
        mark = "  <- 要预测的行" if i >= n_ctx else ""
        lines.append(f"  {i + 1:3d} |{cells} |  {tag}{mark}")
    return "\n".join(lines)


def sample_task_from_scm_prior(rng, n_rows=13, n_feat=4):
    """先验 P：TabPFN 那一类合成先验的简化版。
    随机采一个因果图，每条边给个随机权重，节点值 = 父节点加权和过一个平滑非线性 + 噪声。
    随机挑一个节点当标签、若干节点当观测特征。"""
    n_nodes = n_feat + 3
    # 随机上三角邻接矩阵 = 一个随机 DAG
    adj = (rng.random((n_nodes, n_nodes)) < 0.45) & np.triu(np.ones((n_nodes, n_nodes), bool), 1)
    weights = rng.normal(0, 1.4, (n_nodes, n_nodes)) * adj
    nonlin = rng.choice(["tanh", "sin", "identity"], size=n_nodes)

    vals = np.zeros((n_rows, n_nodes))
    for j in range(n_nodes):
        parents = np.where(adj[:, j])[0]
        z = vals[:, parents] @ weights[parents, j] if len(parents) else np.zeros(n_rows)
        z = z + rng.normal(0, 0.6, n_rows)
        vals[:, j] = {"tanh": np.tanh, "sin": np.sin, "identity": lambda a: a}[nonlin[j]](z)

    # 标签取一个有父节点的靠后的节点，二值化；特征从其余节点里挑
    candidates = [j for j in range(n_nodes) if adj[:, j].any()] or [n_nodes - 1]
    label_node = candidates[-1]
    feat_nodes = rng.choice([j for j in range(n_nodes) if j != label_node], n_feat, replace=False)

    X = vals[:, feat_nodes]
    y = (vals[:, label_node] > np.median(vals[:, label_node])).astype(int)
    structure = [(int(p), int(label_node)) for p in np.where(adj[:, label_node])[0]]
    return X, y, dict(label_node=int(label_node), feat_nodes=feat_nodes.tolist(),
                      edges_into_label=structure, nonlin=str(nonlin[label_node]))


def sample_task_from_rule_prior(rng, n_rows=13, n_feat=4):
    """先验 Q：业务规则型。标签由几条硬阈值规则的与/或决定，而不是平滑的因果关系。
    这是真实业务数据里极常见的形态，但上面那个平滑 SCM 先验从没设想过它。"""
    X = rng.normal(0, 1, (n_rows, n_feat))
    j1, j2 = rng.choice(n_feat, 2, replace=False)
    # 阈值取样本分位数，保证正负样本大致均衡，便于观察
    t1, t2 = np.quantile(X[:, j1], 0.35), np.quantile(X[:, j2], 0.35)
    y = ((X[:, j1] > t1) & (X[:, j2] > t2)).astype(int)
    rule = f"(x{j1 + 1} > {t1:.2f}) 且 (x{j2 + 1} > {t2:.2f})"
    return X, y, dict(rule=rule)


def sample_task_strategic(rng, n_rows=13, n_feat=4):
    """先验 Q'：策略性操纵。标签按原始特征生成，但被判为负的那些行，
    特征会被"改好看"再交上来——模型看到的是改过之后的特征。
    权重取全正，这样"对自己有利的方向"就是把各项特征都调高，现象一眼可见。"""
    X_true = rng.normal(0, 1, (n_rows, n_feat))
    w = np.abs(rng.normal(0, 1, n_feat)) + 0.5
    y = (X_true @ w > 0).astype(int)
    X_seen = X_true.copy()
    losers = np.where(y == 0)[0]
    X_seen[losers] += 1.2 * w / np.linalg.norm(w)
    return X_true, X_seen, y


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


rule("1. 一个「任务」就是一张表。这就是 PFN 的一次输入。")
X, y, meta = sample_task_from_scm_prior(RNG)
print("\n前 10 行带标签（模型可以看到答案），最后 3 行盖住标签让模型猜：\n")
print(fmt_table(X, y, n_query=3))
print("""
注意两件事：
  - 模型看不到列名。没有「年龄」「血压」，只有 x1..x4 这些数字。
  - 它没有语义先验，只有结构先验——关于「数字之间可能存在什么样的关系」的假设。""")

rule("2. 模型实际收到的输入，是一整段序列，不是「训练集 + 测试集」。")
print("""
常规做法是：先用前 10 行训练出一个模型，再拿它去预测后 3 行。两个阶段。

PFN 只有一个阶段。前 10 行（特征+标签）和后 3 行（只有特征）被拼成一段序列，
一次性喂进一个权重固定的 Transformer，前向传播一次，直接吐出后 3 行的答案：

    [x=(-0.13, 0.62, ...), y=1]   <- 这些叫 context，上下文
    [x=( 0.44,-1.05, ...), y=0]
    ...  (共 10 条)
    [x=( 0.91, 0.08, ...), y=?]   <- 这些叫 query，待预测
    [x=(-0.27, 1.33, ...), y=?]

没有梯度更新，没有参数改变。整张表只是它的「输入」，就像一段文字之于对话框。
这也是为什么「往输入里加样本」是你唯一的杠杆——那是你能动的全部东西。""")

rule("3. 「先验」就是那个造表的程序。从同一个先验能采出无穷多张不同的表。")
print("\n下面从同一个程序里连采 3 张表。它们的因果结构各不相同，但共享同一种「风格」：")
for k in range(3):
    Xk, yk, mk = sample_task_from_scm_prior(RNG, n_rows=6, n_feat=3)
    print(f"\n--- 任务 {k + 1} ---")
    print(f"  隐藏的真相：标签由节点 {mk['label_node']} 决定，它的父节点是 "
          f"{[e[0] for e in mk['edges_into_label']] or '无'}，非线性是 {mk['nonlin']}")
    print(fmt_table(Xk, yk))
print("""
TabPFN 就是在几千万张这种表上训练出来的。它学的不是任何一张表，
而是「拿到一张这种风格的表，该怎么从前面的行推断后面的行」这个通用技能。

关键点：这个程序里写了什么假设，模型就带着什么世界观。上面这个程序假设了
  - 变量之间是平滑的关系（tanh / sin，没有断崖）
  - 噪声是高斯的
  - 因果结构是稀疏的
这三条就是它的「先验」。""")

rule("4. 先验错配：你的真实数据不是那个程序造出来的。")
print("\n【错配类型一：业务规则型】现实中标签常常由几条硬阈值规则决定，而不是平滑关系。\n")
Xr, yr, mr = sample_task_from_rule_prior(RNG, n_rows=10, n_feat=4)
print(f"  隐藏的真相：y = 1 当且仅当 {mr['rule']}")
print(fmt_table(Xr, yr))
print("""
  这张表在数值上看起来和上面那些没什么两样——同样是一堆小数加 0/1 标签。
  但生成它的机制是断崖式的阈值，而模型的世界观里只有平滑曲线。
  模型不会报错，它会用平滑曲线去拟合一个阶跃函数，然后自信地给你一个答案。""")

print("\n【错配类型二：策略性操纵】标签按真实特征定，但被拒的人会把特征改好看再提交。\n")
X_true, X_seen, ys = sample_task_strategic(RNG, n_rows=8, n_feat=3)
print("  实际决定标签的原始特征（模型看不到）——三项都高的判 1：")
print(fmt_table(X_true, ys))
print("\n  模型实际看到的（y=0 那几行的特征被本人调高过）：")
print(fmt_table(X_seen, ys))
print("""
  对照两张表：y=1 的行原封不动，y=0 的行每一项都被推高了。
  于是模型看到的数据里，「特征高」反而和「标签为 0」绑在了一起——
  这个关联在真实机制里根本不存在，纯粹是被数据自己的应对行为造出来的。
  造表的那个程序从没设想过「数据会为了应付模型而改变自己」这种事。""")

rule("5. 于是问题就清楚了。")
print("""
模型算出来的，是「假如世界长成造表程序假设的那个样子，答案应该是什么」。
你的数据不来自那个程序，所以它在答另一个世界的题——而且不报错。

三个没人回答的问题：
  (1) 它这时候到底在算什么？错误是随机的，还是有方向、可预测的？
  (2) 能不能在用它之前，只看这张表就算出「这次靠不靠谱」？
  (3) 往输入里加构造出来的样本，能把它的世界观掰回来多少？哪些情况根本掰不动？

第 (3) 问的下半句最值钱：掰不动，就意味着这个场景根本不该用这个模型。
""")
