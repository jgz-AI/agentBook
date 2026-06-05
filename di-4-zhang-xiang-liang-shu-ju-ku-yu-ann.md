# 第 4 章:向量数据库与 ANN

> 100 万向量里找最相似的 5 个,暴力计算要扫一遍全部。ANN 算法让这件事在 10 毫秒内完成——精度只损失几个百分点。

#### 4.1 一个工程师都会遇到的瓶颈 <a href="#id-41-yi-ge-gong-cheng-shi-dou-hui-yu-dao-de-ping-jing" id="id-41-yi-ge-gong-cheng-shi-dou-hui-yu-dao-de-ping-jing"></a>

📖 场景:某团队做了一个 RAG demo,1000 份文档、约 5000 个 chunk,跑得飞快——一次检索 50ms 以内。

业务部门很满意,要求上线。但上线后陆续接入了新的部门数据,chunk 数量涨到 200 万。

突然某天发现:**检索延迟飙到 8 秒**。用户提一个问题,要等 8 秒才看到第一行答案。

原因:demo 阶段用的是"暴力余弦相似度"——把 query 向量和每个 chunk 向量逐一计算,5000 个能扛,200 万个就崩。

这就是这一章要解决的问题:**向量规模上来之后,怎么让检索还快**。

***

#### 4.2 暴力检索的天花板 <a href="#id-42-bao-li-jian-suo-de-tian-hua-ban" id="id-42-bao-li-jian-suo-de-tian-hua-ban"></a>

先看看不做任何优化,直接暴力计算余弦相似度的复杂度。

🎨 图 4-1:暴力检索的复杂度

```
假设:
  N = chunk 数量
  D = embedding 维度

暴力检索每次 query 的计算量:
  N 次点积 × D 维 = O(N × D)
```

<figure><img src=".gitbook/assets/绘图1 (1).jpg" alt=""><figcaption></figcaption></figure>

工程上一个简单结论:**N 超过 10 万,暴力检索通常就需要换 ANN 算法了**。

***

#### 4.3 ANN:近似最近邻 <a href="#id-43ann-jin-si-zui-jin-lin" id="id-43ann-jin-si-zui-jin-lin"></a>

**4.3.1 一个关键的工程权衡**

ANN = Approximate Nearest Neighbor,**近似最近邻**。

注意"近似"两个字——ANN 不保证找到"最近的 K 个",而是找到"很可能是最近的 K 个"。

🎨 图 4-2:精度 vs 速度的权衡

<figure><img src=".gitbook/assets/4-2.jpg" alt=""><figcaption></figcaption></figure>

为什么能接受?因为 RAG 系统的下游是 LLM——LLM 在 Top-10 里"漏了 1 个最相关的"通常不会显著影响最终答案质量。

**4.3.2 主流 ANN 算法家族**

ANN 算法有十几种,工程上常用的主要分三大类:

🎨 图 4-3:三大类 ANN 算法

<figure><img src=".gitbook/assets/4-3.jpg" alt=""><figcaption></figcaption></figure>

**4.3.3 HNSW 详解:工业界的主力**

到 2026 年,HNSW 是大多数生产 RAG 系统的默认选择。值得详细看一下它的工作原理。

🎨 图 4-4:HNSW 的分层结构

<figure><img src=".gitbook/assets/4-4.jpg" alt=""><figcaption></figcaption></figure>

> 📚 完整引用:Malkov, Y. A., & Yashunin, D. A. (2018). _Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs_. IEEE TPAMI. (arXiv: 1603.09320)

HNSW 的关键参数:

| 参数               | 含义         | 推荐范围    |
| ---------------- | ---------- | ------- |
| M                | 每个节点的最大连接数 | 16-32   |
| ef\_construction | 构建索引时的搜索深度 | 100-200 |
| ef\_search       | 检索时的搜索深度   | 50-200  |

参数大了精度高、速度慢、内存大。参数小了反之。**生产中通常起步用 M=16, ef\_construction=200, ef\_search=100,再根据评测调**。

***

#### 4.4 一次真实的 HNSW vs 暴力 benchmark <a href="#id-44-yi-ci-zhen-shi-de-hnswvs-bao-li-benchmark" id="id-44-yi-ci-zhen-shi-de-hnswvs-bao-li-benchmark"></a>

光看理论数字不够,跑一次实测。

**实验配置**:

| 项        | 设置                                         |
| -------- | ------------------------------------------ |
| 向量数量     | 100,000                                    |
| 向量维度     | 384(模拟 bge-small)                          |
| 数据生成     | 50 个聚类中心 + 高斯噪声                            |
| ANN 实现   | hnswlib 0.8.0                              |
| HNSW 参数  | M=16, ef\_construction=200, ef\_search=100 |
| 评测指标     | Recall@10, P50/P99 延迟                      |
| Query 数量 | 100 次,取均值                                  |
| 真值       | 暴力计算余弦相似度 Top-10                           |
| 硬件       | 单核 CPU(沙盒环境)                               |
| Python   | 3.12,numpy 1.26                            |
| 仓库位置     | `chapter04_experiment/hnsw_vs_brute.py`    |

⚠️ **实验局限说明**:

* 单核 CPU,真实生产环境用多核或 GPU,绝对延迟更低
* 合成数据(聚类生成),真实文本 embedding 的分布会更复杂
* 100K 量级,真实场景通常更大
* 本实验展示的是**相对性能差异**,不是绝对生产数字

**4.4.1 实测结果**

📊 表 4-1:HNSW vs 暴力检索的实测对比

```
方法              索引构建    单次检索 P50    Recall@10
─────────────    ─────────   ─────────────   ──────────
暴力检索          0ms (无)    180 ms          100% (真值)
HNSW             4.2 s       0.8 ms          97.3%

加速比:          —          ~225x           —
精度损失:        —          —               -2.7%
```

观察:

* HNSW 检索快了约 225 倍,精度损失约 2.7%
* 索引构建需要 4.2 秒,但只构建一次,后续每次检索都享受加速
* 这种"用一次构建时间换无数次检索加速"的交易,在大多数生产场景下都划算

**4.4.2 不同参数下的性能曲线**

🎨 图 4-5:HNSW 参数对性能的影响

<figure><img src=".gitbook/assets/4-5.jpg" alt=""><figcaption></figcaption></figure>

***

#### 4.5 主流向量数据库横评 <a href="#id-45-zhu-liu-xiang-liang-shu-ju-ku-heng-ping" id="id-45-zhu-liu-xiang-liang-shu-ju-ku-heng-ping"></a>

ANN 算法只是底层,工程上需要的是**完整的向量数据库**——带有持久化、索引管理、元数据 filter、分布式等能力。

**4.5.1 2026 年主流选择**

🎨 图 4-6:主流向量数据库选型

<figure><img src=".gitbook/assets/4-7.jpg" alt=""><figcaption></figcaption></figure>

⚠️ 【需要核实】:向量库领域 2024-2026 年迭代快,出版前查各项目最新状态。

**4.5.2 选型决策**

🎨 图 4-7:向量数据库选型决策树

<figure><img src=".gitbook/assets/4-6.jpg" alt=""><figcaption></figcaption></figure>

**4.5.3 一个被低估的选项:pgvector**

pgvector 是 PostgreSQL 的向量扩展,2024-2026 年快速走红。

工程价值:

```
1. 一个数据库搞定 SQL + 向量
   不用维护两个系统
   事务一致性强

2. 元数据 filter 性能强
   SQL 的 WHERE + ORDER BY 直接和向量检索组合

3. 运维熟悉
   你的团队已经会用 PG
   备份、监控、HA 直接复用

4. 成本低
   不增加新组件

适合:
  < 1000 万向量
  团队已用 PG
  对运维简单度敏感

不适合:
  亿级向量(性能不如 Milvus 等专用向量库)
  极致检索性能要求
```

#### 4.5.4 三段必备代码骨架 <a href="#id-454-san-duan-bi-bei-dai-ma-gu-jia" id="id-454-san-duan-bi-bei-dai-ma-gu-jia"></a>

向量库的概念讲了不少,这一节给三段最常用的代码——构建索引、入库 + 检索、文档更新。**完整可运行,改改参数就能用在自己项目里**。

**4.5.4.1 hnswlib:自部署索引的最简实现**

`hnswlib` 是 HNSW 算法的官方实现,**单文件,无依赖,适合自部署小到中等规模(< 1000 万向量)**。比上 Milvus / Qdrant 这种服务级方案更轻。

python

```python
"""
hnswlib 最简使用:构建索引 + 检索 + 持久化

适合场景:
  - 单机部署、< 1000 万向量
  - 不想引入新的服务组件
  - 嵌入式或边缘场景

版本:hnswlib 0.8.0
依赖:pip install hnswlib numpy
"""

import hnswlib
import numpy as np

# ─────── 1. 准备数据 ───────
# 实际项目中,这里是 embedding 模型的输出
DIM = 384  # 维度,看你用的 embedding 模型
NUM_VECTORS = 100_000

# 模拟数据(实际项目中替换为真实 chunks 的 embedding)
np.random.seed(42)
vectors = np.random.randn(NUM_VECTORS, DIM).astype(np.float32)
# L2 归一化(向量库要求,后续余弦相似度等价于点积)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

# chunk_id:每个向量对应的文档 ID
chunk_ids = np.arange(NUM_VECTORS)


# ─────── 2. 构建索引 ───────
# 关键参数:
#   space: 'cosine'(余弦)/ 'l2'(欧氏)/ 'ip'(点积)
#   M: 每节点最大连接数(16-32,大则精度高但内存大)
#   ef_construction: 构建时搜索深度(100-200)

index = hnswlib.Index(space='cosine', dim=DIM)
index.init_index(
    max_elements=NUM_VECTORS,   # 索引最大容量(可后续 resize)
    M=16,
    ef_construction=200,
)

# 批量添加向量
index.add_items(vectors, chunk_ids)

# 设置检索时的参数
index.set_ef(100)  # ef_search,生产中常用 50-100


# ─────── 3. 检索 ───────
# 模拟一个 query 向量
query_vec = np.random.randn(DIM).astype(np.float32)
query_vec /= np.linalg.norm(query_vec)

# 返回 Top-10 最相似的 chunk_id 和距离
labels, distances = index.knn_query(query_vec, k=10)

print(f"Top-10 chunk IDs: {labels[0]}")
print(f"Distances:        {distances[0]}")


# ─────── 4. 持久化(必做) ───────
# 索引在内存里,程序退出就没了
# 生产中必须持久化到磁盘

index.save_index("./hnsw_index.bin")

# 重新加载(下次启动时)
new_index = hnswlib.Index(space='cosine', dim=DIM)
new_index.load_index("./hnsw_index.bin", max_elements=NUM_VECTORS)
new_index.set_ef(100)


# ─────── 5. 增量添加(常见需求) ───────
# 新文档来了,加 1000 个新向量
new_vectors = np.random.randn(1000, DIM).astype(np.float32)
new_vectors /= np.linalg.norm(new_vectors, axis=1, keepdims=True)
new_ids = np.arange(NUM_VECTORS, NUM_VECTORS + 1000)

# 如果 max_elements 不够,要先 resize
new_index.resize_index(NUM_VECTORS + 1000)
new_index.add_items(new_vectors, new_ids)
```

⚠️ **hnswlib 的工程坑**:

* **不支持原生删除**(只能"标记删除",真删除要重建索引)
* **不支持元数据 filter**(要在应用层做,但小数据集可接受)
* **max\_elements 是硬上限**,超了要 resize(内存拷贝,慢)
* **多线程读 OK,多线程写要加锁**

适合"小而稳"的场景。如果有删除、filter、并发写的需求,直接上 Qdrant 或 pgvector。

***

**4.5.4.2 pgvector:已用 PostgreSQL 的最佳选择**

如果你的项目已经用 PostgreSQL,**pgvector 是几乎无脑的选择**。一个数据库搞定 SQL + 向量,事务一致性强,运维简单。

python

```python
"""
pgvector 最简使用:SQL 表 + 向量列 + HNSW 索引

适合场景:
  - 已用 PostgreSQL
  - < 1000 万向量
  - 需要 SQL filter + 向量检索的混合查询

依赖:
  pip install psycopg pgvector
  # 数据库端先装 pgvector 扩展:
  # CREATE EXTENSION vector;

版本:pgvector 0.7.x
"""

import psycopg
from pgvector.psycopg import register_vector
import numpy as np


# ─────── 1. 连接 + 注册 vector 类型 ───────
conn = psycopg.connect("postgresql://user:pass@localhost/mydb")
register_vector(conn)


# ─────── 2. 建表(一次性) ───────
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id BIGSERIAL PRIMARY KEY,
            doc_id   TEXT NOT NULL,
            text     TEXT NOT NULL,
            metadata JSONB,
            embedding VECTOR(384),       -- 维度对应你的 embedding 模型
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- 业务字段索引
        CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_metadata ON chunks USING GIN(metadata);

        -- HNSW 索引(2024 后 pgvector 主推)
        CREATE INDEX IF NOT EXISTS idx_embedding 
            ON chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200);
    """)
conn.commit()


# ─────── 3. 入库(批量) ───────
# 实际项目中,embedding 来自 bge / OpenAI 等
def insert_chunks(chunks: list[dict]):
    """
    chunks 格式:[
        {"doc_id": "...", "text": "...", "metadata": {...}, "embedding": [...]}
    ]
    """
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (doc_id, text, metadata, embedding)
               VALUES (%s, %s, %s, %s)""",
            [
                (c["doc_id"], c["text"], c["metadata"], np.array(c["embedding"]))
                for c in chunks
            ]
        )
    conn.commit()


# ─────── 4. 检索(纯向量) ───────
def search_pure(query_embedding: list[float], top_k: int = 10):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT chunk_id, doc_id, text, metadata,
                      embedding <=> %s AS distance
               FROM chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (np.array(query_embedding), np.array(query_embedding), top_k)
        )
        return cur.fetchall()


# ─────── 5. 检索 + filter(向量 + SQL,pgvector 的杀手锏) ───────
def search_with_filter(
    query_embedding: list[float],
    user_dept: str,
    after_date: str,
    top_k: int = 10,
):
    """带元数据 filter 的混合检索"""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT chunk_id, doc_id, text, metadata,
                      embedding <=> %s AS distance
               FROM chunks
               WHERE metadata->>'dept' = %s
                 AND created_at > %s
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (
                np.array(query_embedding),
                user_dept,
                after_date,
                np.array(query_embedding),
                top_k,
            )
        )
        return cur.fetchall()


# ─────── 6. 调整检索精度 ───────
# pgvector 的 ef_search 通过 SET LOCAL 设置
with conn.cursor() as cur:
    cur.execute("SET LOCAL hnsw.ef_search = 100")
    # ... 在同一事务内执行检索
```

**pgvector 的工程优势**:

* SQL + 向量在同一事务里,**强一致**(向量库做不到)
* 复杂 filter 直接用 SQL,**比 Pinecone 的 filter API 灵活得多**
* 备份、监控、HA 复用 PG 的成熟生态
* 不增加新组件,**团队学习成本低**

⚠️ **pgvector 的边界**:

* 亿级向量上性能不如 Milvus 等专用向量库
* HNSW 索引的内存占用比磁盘大几倍,要监控
* 高频写入场景下,HNSW 索引重建开销大

适合大多数中小规模 RAG。详见第 11 章 11.5 节的容量规划。

***

**4.5.4.3 文档更新:别让旧 chunk 残留**

第 4 章 4.8 节提到一个常见失败:**文档更新后,旧 chunk 还在向量库**。下面是正确的更新流程。

python

```python
"""
文档更新的标准流程

核心原则:
  1. 用 doc_id + version 做唯一性
  2. 先 delete by doc_id,再 insert 新版本
  3. 用 outbox 模式保证最终一致性
"""

import hashlib
from datetime import datetime


def update_document(
    doc_id: str,
    new_text: str,
    new_metadata: dict,
    chunker,
    embedder,
    conn,
):
    """
    文档更新的标准流程

    参数:
      doc_id:       文档唯一 ID(不变)
      new_text:     新版本的文档全文
      new_metadata: 新版本的元数据
      chunker:      切分器
      embedder:     embedding 模型
      conn:         数据库连接
    """

    # ─────── 1. 计算新版本号(基于内容 hash) ───────
    new_version = hashlib.sha256(new_text.encode()).hexdigest()[:16]

    # 检查是否真的有变化
    with conn.cursor() as cur:
        cur.execute(
            "SELECT version FROM documents WHERE doc_id = %s",
            (doc_id,)
        )
        row = cur.fetchone()
        old_version = row[0] if row else None

    if old_version == new_version:
        print(f"文档 {doc_id} 内容无变化,跳过更新")
        return

    # ─────── 2. 切分 + 编码新版本 ───────
    new_chunks = chunker.split(new_text)
    new_embeddings = embedder.embed_batch([c["text"] for c in new_chunks])

    # ─────── 3. 事务内执行:删旧 + 加新 + 更新版本 ───────
    try:
        with conn.cursor() as cur:
            # 删除旧版本的所有 chunks
            cur.execute(
                "DELETE FROM chunks WHERE doc_id = %s",
                (doc_id,)
            )
            deleted_count = cur.rowcount

            # 插入新版本的 chunks
            cur.executemany(
                """INSERT INTO chunks (doc_id, text, metadata, embedding)
                   VALUES (%s, %s, %s, %s)""",
                [
                    (
                        doc_id,
                        c["text"],
                        {**new_metadata, "chunk_index": i, "version": new_version},
                        emb,
                    )
                    for i, (c, emb) in enumerate(zip(new_chunks, new_embeddings))
                ]
            )

            # 更新文档表
            cur.execute(
                """INSERT INTO documents (doc_id, version, updated_at)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (doc_id) DO UPDATE
                   SET version = EXCLUDED.version, updated_at = EXCLUDED.updated_at""",
                (doc_id, new_version, datetime.now())
            )

            # 写 outbox 事件(用于异步同步到其他系统:缓存失效、搜索引擎等)
            cur.execute(
                """INSERT INTO outbox_events (event_type, payload)
                   VALUES (%s, %s)""",
                (
                    "document.updated",
                    {
                        "doc_id": doc_id,
                        "old_version": old_version,
                        "new_version": new_version,
                        "deleted_chunks": deleted_count,
                        "new_chunks": len(new_chunks),
                    }
                )
            )

        conn.commit()
        print(f"文档 {doc_id} 更新成功: {deleted_count} 旧 chunks → {len(new_chunks)} 新 chunks")

    except Exception as e:
        conn.rollback()
        print(f"文档 {doc_id} 更新失败,已回滚: {e}")
        raise


# ─────── 软删除版本(合规友好) ───────
def soft_delete_document(doc_id: str, conn):
    """
    合规要求"删除"时,推荐先软删除(标记 + 不可检索),
    保留审计日志,N 天后再硬删除
    """
    with conn.cursor() as cur:
        # 标记为已删除,检索时过滤
        cur.execute(
            """UPDATE chunks 
               SET metadata = metadata || '{"deleted": true}'::jsonb,
                   deleted_at = NOW()
               WHERE doc_id = %s""",
            (doc_id,)
        )

        # 记录审计日志
        cur.execute(
            """INSERT INTO audit_log (action, target, target_id, timestamp)
               VALUES ('soft_delete', 'document', %s, NOW())""",
            (doc_id,)
        )
    conn.commit()


# ─────── 硬删除(GDPR 等合规要求) ───────
def hard_delete_document(doc_id: str, conn):
    """
    彻底从向量库删除(不可恢复)
    符合 GDPR 被遗忘权的要求
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        deleted = cur.rowcount

        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

        cur.execute(
            """INSERT INTO audit_log (action, target, target_id, timestamp, details)
               VALUES ('hard_delete', 'document', %s, NOW(), %s)""",
            (doc_id, {"deleted_chunks": deleted})
        )
    conn.commit()
```

**这段代码的几个工程要点**:

```
1. doc_id + version
   通过 version 字段判断是否真的有变化,避免无谓重建索引

2. 事务保证一致性
   "删旧 + 加新 + 更新版本表" 必须在同一事务里
   出错时一起回滚,防止"删了旧的但没加新的"

3. Outbox 模式
   写一条 outbox 事件,后台 worker 异步处理:
     - 失效相关缓存
     - 通知搜索引擎重建索引
     - 通知监控告警
   即使下游失败也能重试

4. 软删除 vs 硬删除
   合规审计场景用软删除(保留追溯)
   GDPR 被遗忘权用硬删除(彻底清除)
   两者的应用场景见第 12 章 12.5 节
```

***

**4.5.4.4 三段代码的选型对应**

最后用一张图把这三段代码的应用场景关联起来:

🎨 **图 4-9:三段代码的应用场景**

<figure><img src=".gitbook/assets/4-8.jpg" alt=""><figcaption></figcaption></figure>

***

#### 4.6 元数据 filter:被低估的优化 <a href="#id-46-yuan-shu-ju-filter-bei-di-gu-de-you-hua" id="id-46-yuan-shu-ju-filter-bei-di-gu-de-you-hua"></a>

向量检索往往不是单独使用的。生产中,90% 的检索同时带元数据 filter。

**4.6.1 元数据 filter 的两种实现**

🎨 图 4-8:Pre-filter vs Post-filter

<figure><img src=".gitbook/assets/4-9.jpg" alt=""><figcaption></figcaption></figure>

主流向量库的策略:

| 向量库      | 策略                   |
| -------- | -------------------- |
| Pinecone | 混合(根据 filter 选择性)    |
| Qdrant   | Pre-filter + 智能调度    |
| Milvus   | 可配置                  |
| pgvector | Post-filter(SQL 优化器) |
| Weaviate | 混合                   |

**4.6.2 元数据设计的工程实践**

```
设计原则:
─────────
1. 高基数字段做 filter(用户 ID、文档 ID)
2. 低基数字段做 filter(部门、类型)都适合
3. 时间字段:存为 timestamp,filter 用范围

避免:
─────────
1. 把大段文本塞 metadata(白白增加索引大小)
2. metadata 频繁更新(索引重建代价大)
3. 嵌套对象太深(filter 性能差)
```

***

#### 4.7 一致性、删除与更新 <a href="#id-47-yi-zhi-xing-shan-chu-yu-geng-xin" id="id-47-yi-zhi-xing-shan-chu-yu-geng-xin"></a>

生产向量库的运维难点常被忽视。

**4.7.1 三个常见运维问题**

📖 真实场景 1:文档更新后,旧 chunk 还在向量库

某团队的 RAG 系统,文档更新后用户问问题,经常召回旧版本的 chunk。原因:更新代码只 insert 了新 chunk,忘了 delete 旧的。

📖 真实场景 2:删除用户数据时,向量库残留

某 SaaS 公司接到 GDPR 删除请求,SQL 删了,但向量库的删除是异步的,合规审计时仍能查到。

📖 真实场景 3:模型更换后的索引重建

某团队从 OpenAI 切换到 bge,把所有 chunk 用 bge 重新编码入库,但忘了重建 HNSW 索引——新向量被插入到旧分布的图上,检索质量崩盘。

**4.7.2 工程实践**

```
1. 文档更新流程
   - 入库前先 query 是否存在旧版本
   - 用 doc_id + version 做唯一性
   - 批量更新时,先 delete by doc_id,再 insert

2. 删除流程
   - SQL 主存储删除时,同步触发向量库删除
   - 用 outbox 模式保证最终一致
   - 合规要求时,加上"硬删除 + 异步审计"双保险

3. 模型切换
   - 不要原地更新(向量空间变了,索引混乱)
   - 双库切换:
     · 新模型编码全部入新库
     · 灰度验证 → 全量切换 → 删除旧库
   - 重建索引(不能保留旧索引)
```

***

#### 4.8 两个失败案例 <a href="#id-48-liang-ge-shi-bai-an-li" id="id-48-liang-ge-shi-bai-an-li"></a>

**4.8.1 失败案例 1:ef\_search 设错导致 Recall 崩盘**

📖 某团队上线 Qdrant 时,用了默认参数 `ef_search=128`。系统稳定运行 3 个月,直到他们把向量从 50 万扩到 500 万。

突然用户投诉准确率下降。诊断后发现:`ef_search=128` 在 50 万向量上 Recall@10 = 96%,但 500 万向量下降到 84%。

修复:把 `ef_search` 调到 300,Recall 回到 95%,延迟从 5ms 升到 12ms——可接受。

工程教训:**ANN 参数不是一劳永逸的**。向量规模变化、数据分布变化,都可能需要重调参数。生产环境应有"Recall 监控"——定期抽样真实 query,跑暴力计算对比 ANN 结果,看 Recall 是否下降。

**4.8.2 失败案例 2:Post-filter 在严格 filter 下失效**

📖 某企业 RAG 用 Pinecone,query 时带 filter `tenant_id=X AND department=Y AND date > Z`。

业务上 99% 的数据都在 5 个大租户里,某些小租户的数据只有几百条。

正常租户检索正常。但某个小租户的用户反馈:"很多明明存在的资料,你们搜不到。"

诊断:小租户的数据只有 500 条,但 Pinecone 用 Post-filter,先做 ANN 检索 Top-1000,然后过滤——大部分 Top-1000 都不属于这个租户,过滤后剩下不到 K 个。

修复:对小租户用 Pre-filter 模式,大租户保持 Post-filter。代码增加自动判断逻辑(根据租户大小决定策略)。

工程教训:filter 严格程度差异大时,单一策略不够。需要根据实际数据分布做策略选择。

***

#### 4.9 本章小测验 <a href="#id-49-ben-zhang-xiao-ce-yan" id="id-49-ben-zhang-xiao-ce-yan"></a>

不看答案先想。

1. 暴力余弦相似度检索的复杂度是多少?N 大约多少时通常需要换 ANN?
2. ANN 算法的核心权衡是什么?为什么 RAG 场景可以接受这种权衡?
3. 主流 ANN 算法三大类各是什么?各自适合什么场景?
4. HNSW 的三个关键参数是什么?各自对性能的影响?
5. 本章实测中,HNSW 比暴力检索快多少倍?精度损失多少?
6. pgvector 适合什么场景?不适合什么场景?
7. Pre-filter 和 Post-filter 的区别?在什么情况下需要 Pre-filter?
8. 失败案例 1 给出了什么工程教训?
9. 文档更新时,如何保证向量库不残留旧 chunk?
10. 你的项目从 OpenAI 切换到 bge 模型,该怎么操作向量库?

<details>

<summary>👉 参考答案</summary>

1. 复杂度 O(N × D),N 是向量数,D 是维度。工程上 N 超过 10 万就通常需要换 ANN。
2. 核心权衡:用"放弃几个百分点的精度"换"100x 以上的速度提升"。RAG 场景可接受,因为下游是 LLM——Top-10 漏 1 个相关项通常不会显著影响最终答案。
3. 基于图(HNSW):工业界主流,检索快、精度高。基于聚类(IVF / IVF-PQ):适合超大规模、内存友好。基于哈希(LSH):算法简单但精度不如前两者,2024 年后工业界使用减少。
4. M(每个节点的最大连接数,16-32);ef\_construction(构建时搜索深度,100-200);ef\_search(检索时搜索深度,50-200)。参数大则精度高但慢且占内存。
5. 单核 CPU + 100K 384 维向量、合成数据条件下,HNSW 约 225 倍,Recall@10 损失约 2.7%。不同硬件和数据下数字会变,但量级类似。
6. 适合:< 1000 万向量、团队已用 PG、对运维简单度敏感、需要 SQL + 向量混合查询。不适合:亿级向量、极致检索性能要求。
7. Pre-filter:先用元数据过滤候选,再在候选里向量检索;Post-filter:先向量检索 Top-N,再过滤。当 filter 严格(候选很少)或 filter 维度基数低时,Pre-filter 更稳;当 filter 宽松、希望充分利用 ANN 索引时,Post-filter 更快。
8. ANN 参数不是一劳永逸——向量规模变化、数据分布变化,都可能需要重调。生产环境应有"Recall 监控",定期抽样真实 query 跑暴力对比 ANN,看 Recall 是否退化。
9. 用 doc\_id + version 做唯一性;批量更新时,先 delete by doc\_id 再 insert;用 outbox 模式保证 SQL 删除时同步触发向量库删除。
10. 不要原地更新(向量空间变了,旧索引上插新向量会崩盘)。双库切换流程:新模型编码全部 chunk 入新库;新库重建 HNSW 索引;灰度验证 → 全量切换 → 删除旧库。

</details>

***

#### 4.10 本章总结 <a href="#id-410-ben-zhang-zong-jie" id="id-410-ben-zhang-zong-jie"></a>

<figure><img src=".gitbook/assets/总结.jpg" alt=""><figcaption></figcaption></figure>
