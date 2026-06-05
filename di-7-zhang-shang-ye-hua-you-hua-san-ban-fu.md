# 第 7 章:商业化优化三板斧

> 从 62% 准确率到 92%,中间不靠魔法,靠三件事:多路召回、Reranker、Query 改写。这一章给数据。

#### 7.1 一个真实的优化轨迹 <a href="#id-71-yi-ge-zhen-shi-de-you-hua-gui-ji" id="id-71-yi-ge-zhen-shi-de-you-hua-gui-ji"></a>

📖 场景:某 SaaS 公司的客服 RAG,上线一个月,准确率卡在 62%。产品反馈"AI 经常答不到点子上"。

团队花 3 周做了三轮优化:

```
基线:          Naive RAG(向量检索 + Top-3)
                准确率 62%

优化 1:        +多路召回(BM25 + 向量)
                准确率 74%   (+12%)

优化 2:        +Reranker(bge-reranker-v2-m3)
                准确率 84%   (+10%)

优化 3:        +Query 改写(HyDE)
                准确率 92%   (+8%)

总耗时:        3 周
代码改动:      约 300 行
延迟变化:      400ms → 1200ms
成本变化:      每次 query 成本 +60%
```

这就是这一章的主题:**用三板斧把 RAG 准确率从"勉强能用"推到"商业可用"**。三个技巧第 6 章都讲过,这一章讲它们组合使用的工程实操和真实数据。

***

#### 7.2 三板斧总览 <a href="#id-72-san-ban-fu-zong-lan" id="id-72-san-ban-fu-zong-lan"></a>

🎨 图 7-1:三板斧的协同关系

<figure><img src=".gitbook/assets/绘图1_backup_01174.jpg" alt=""><figcaption></figcaption></figure>

**为什么是这三个?** 因为它们各自解决 RAG 不同环节的"系统性偏差":

```
板斧 1 解决 query 端的偏差
板斧 2 解决检索端的偏差
板斧 3 解决排序端的偏差
```

三个偏差独立,所以三板斧的收益**可以叠加**——这是它们 ROI 高的根本原因。

***

#### 7.3 板斧 1:多路召回 <a href="#id-73-ban-fu-1-duo-lu-zhao-hui" id="id-73-ban-fu-1-duo-lu-zhao-hui"></a>

**7.3.1 为什么是 ROI 最高的**

按工程经验,**多路召回通常是三板斧里 ROI 最高的**。原因:

```
1. 改动小
   只需要再跑一次 BM25 检索,然后 RRF 融合

2. 收益稳定
   在几乎所有 query 风格下都有提升
   不像 Reranker 在某些场景反而降低

3. 代价可控
   BM25 在 Elasticsearch / OpenSearch 上几乎免费
   延迟增加几十毫秒(可并行)
```

**7.3.2 BM25 + Vector 的工程实现**

python

```python
"""
多路召回的最简实现:BM25 + Vector + RRF

无论你用什么向量库,这段代码的逻辑都通用
"""

from rank_bm25 import BM25Okapi
import numpy as np

def chinese_bigram(text: str) -> list[str]:
    """中文字符 bigram tokenizer"""
    return [text[i:i+2] for i in range(len(text) - 1)]


class HybridRetriever:
    def __init__(self, chunks: list[dict]):
        """
        chunks 格式:[
            {"chunk_id": "...", "text": "...", "embedding": [...]}
        ]
        """
        self.chunks = chunks

        # BM25 索引
        self.bm25 = BM25Okapi([chinese_bigram(c["text"]) for c in chunks])

        # 向量索引(简化:存在内存,实际用 pgvector / Qdrant 等)
        self.embeddings = np.array([c["embedding"] for c in chunks])
        # L2 归一化
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        bm25_k: int = 50,
        vector_k: int = 50,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Hybrid 检索:BM25 召回 + Vector 召回 + RRF 融合

        bm25_k / vector_k:各路召回的数量(通常 50-100)
        top_k:           最终返回数量
        rrf_k:           RRF 平滑参数(防止 rank=1 时分数过大)
        """
        # 1. BM25 检索
        bm25_scores = self.bm25.get_scores(chinese_bigram(query))
        bm25_top = np.argsort(-bm25_scores)[:bm25_k]

        # 2. Vector 检索
        q_emb = np.array(query_embedding)
        q_emb /= np.linalg.norm(q_emb)
        vector_scores = self.embeddings @ q_emb
        vector_top = np.argsort(-vector_scores)[:vector_k]

        # 3. RRF 融合(用 rank 而非 score)
        rrf_scores = {}
        for rank, idx in enumerate(bm25_top, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)
        for rank, idx in enumerate(vector_top, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)

        # 4. 返回 Top-K
        top_idx = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])[:top_k]
        return [self.chunks[i] for i in top_idx]
```

**7.3.3 工程上的几个调优经验**

```
1. bm25_k 和 vector_k 设多少
   常见取值:50-100
   太小:RRF 融合空间小,效果差
   太大:Reranker 阶段负担重

2. RRF 的 k 参数
   默认 60(原始论文的推荐值)
   生产中很少需要调

3. BM25 的 tokenizer 选择
   中文:bigram 或 jieba 分词
   bigram 在大多数场景下够用,且无外部依赖

4. 是否在两路上各做 Reranker
   不建议
   Reranker 应该在 RRF 融合后只跑一次
```

**7.3.4 一个失败案例:RRF 参数被忽略**

📖 某团队上了 Hybrid 检索后,准确率只涨了 3 个百分点,远低于预期。

诊断后发现:

* 他们用的库默认 `bm25_k = vector_k = 5`(只融合 Top-5)
* Top-5 里 BM25 和 Vector 重合度很高,RRF 几乎没融合到不同的结果
* 改成 `bm25_k = vector_k = 50` 后,准确率涨了 11 个百分点

教训:**用 Hybrid 检索时,各路召回的 K 必须比最终输出大几倍**,这样 RRF 才有融合空间。

***

#### 7.4 板斧 2:Reranker <a href="#id-74-ban-fu-2reranker" id="id-74-ban-fu-2reranker"></a>

**7.4.1 配合多路召回使用**

第 6 章已经详细讲过 Reranker。这里强调它和多路召回的**协同关系**:

```
没有 Reranker:
─────────
Hybrid 召回 → Top-5 → 喂 LLM
问题:Top-5 中前后顺序不一定准

有 Reranker:
─────────
Hybrid 召回 → Top-50 → Reranker 精排 → Top-5 → 喂 LLM
                       ↑
                       Cross-encoder 重新打分
```

**Reranker 的真实价值在于它能消化更多候选**——从 50 个里精排,远比从 5 个里直接挑要稳。

**7.4.2 Reranker 的工程实现**

python

```python
"""
用 bge-reranker 做精排
"""

from FlagEmbedding import FlagReranker

class RerankerService:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # use_fp16: 推理速度提升约 2 倍,精度损失可忽略
        self.reranker = FlagReranker(model_name, use_fp16=True)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        candidates 格式:[{"chunk_id": "...", "text": "..."}, ...]

        返回 Top-K(按 reranker 分数降序)
        """
        if not candidates:
            return []

        # Cross-encoder 同时编码 (query, doc) 对
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.compute_score(pairs, normalize=True)

        # 按分数降序排
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: -x[1])

        # 在 metadata 里记录 rerank 分数,便于调试
        result = []
        for c, s in scored[:top_k]:
            c_with_score = {**c, "rerank_score": float(s)}
            result.append(c_with_score)
        return result
```

**7.4.3 Reranker 的常见误用**

📖 **失败案例:把 Reranker 当成 Recall 工具**

某团队听说"Reranker 能提升准确率",在第一阶段不做多路召回,**直接用 Reranker 给全库打分**。

python

```python
# 错误做法
all_chunks = vector_db.get_all()  # 拉全库 1 万个 chunks
scored = reranker.compute_score([(query, c.text) for c in all_chunks])
top_5 = ...
```

结果:每个 query 要跑 1 万次 Cross-encoder 推理,**单次 query 延迟 30 秒**。

修复:Reranker 必须放在召回之后,候选数量控制在 50-100。

教训:**Reranker 是精排工具,不是召回工具**。

***

#### 7.5 板斧 3:Query 改写 <a href="#id-75-ban-fu-3query-gai-xie" id="id-75-ban-fu-3query-gai-xie"></a>

第 6 章讲过 Query 改写的三种方式(同义词扩展、HyDE、Multi-Query)。在三板斧中,Query 改写的 ROI **最看场景**:

```
高 ROI 场景:
─────────
- 用户 query 风格多样、口语化
- 同义词体系复杂的领域(医疗、法律、学术)
- 用户和文档作者不是同一群人(例如客户 vs 内部专家)

低 ROI 场景:
─────────
- 用户 query 高度规范(BI 报表类、技术文档查询)
- 用户就是文档的作者(内部知识库)
- 延迟敏感(每次多 1 次 LLM 调用)
```

**7.5.1 三种改写的选择**

```
方式 1:同义词扩展
─────────
"礼拜六" → "礼拜六 / 周末 / 休息日"
最简单,延迟最低
适合:领域同义词不多、改写规则可枚举

方式 2:HyDE
─────────
让 LLM 假设答案,用假设的答案文本去检索
精度高,延迟中等(1 次 LLM 调用)
适合:用户 query 简短、文档表达详细

方式 3:Multi-Query
─────────
生成 N 个不同表达的 query,分别检索后合并
精度最高,延迟最高(1 次 LLM 调用 + N 次检索)
适合:对 Recall 极敏感的场景
```

工程上通常**从 HyDE 起步**——一次 LLM 调用,效果稳定,工程复杂度低。

***

#### 7.6 三板斧的完整流程 <a href="#id-76-san-ban-fu-de-wan-zheng-liu-cheng" id="id-76-san-ban-fu-de-wan-zheng-liu-cheng"></a>

python

```python
"""
完整的三板斧 RAG pipeline
"""

class CommercialRAG:
    def __init__(self, retriever, reranker, llm, embedder):
        self.retriever = retriever  # HybridRetriever(板斧 2)
        self.reranker = reranker    # RerankerService(板斧 3)
        self.llm = llm
        self.embedder = embedder

    def answer(self, query: str) -> dict:
        # ─── 板斧 1:Query 改写(HyDE) ───
        hyde_text = self._hyde_rewrite(query)

        # 同时检索:用 HyDE 文本 + 原 query
        # 提升对短 query 的鲁棒性
        hyde_emb = self.embedder.embed(hyde_text)

        # ─── 板斧 2:Hybrid 召回 ───
        candidates = self.retriever.search(
            query=query,
            query_embedding=hyde_emb,
            top_k=50,
            bm25_k=50,
            vector_k=50,
        )

        # ─── 板斧 3:Reranker 精排 ───
        top_chunks = self.reranker.rerank(query, candidates, top_k=5)

        # ─── 喂 LLM ───
        answer = self._generate(query, top_chunks)

        return {
            "answer": answer,
            "chunks": top_chunks,
            "hyde_text": hyde_text,
        }

    def _hyde_rewrite(self, query: str) -> str:
        prompt = f"""假设你已经知道这个问题的答案,用 150 字写一段回答。
回答应该像文档原文一样客观、详细。

问题:{query}

假设回答:"""
        return self.llm.generate(prompt)

    def _generate(self, query: str, chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
        )
        prompt = f"""基于资料回答问题,资料外的内容不要补充。
答案末尾用 [1][2] 标注引用编号。

资料:
{context}

问题:{query}

答案:"""
        return self.llm.generate(prompt)
```

***

#### 7.7 三板斧的真实效果分解 <a href="#id-77-san-ban-fu-de-zhen-shi-xiao-guo-fen-jie" id="id-77-san-ban-fu-de-zhen-shi-xiao-guo-fen-jie"></a>

回到开头那个 62% → 92% 的故事,看看每一板斧的贡献是怎么来的。

📊 表 7-1:三板斧分解的真实数据(基于业界常见模式归纳)

```
配置                         准确率   增量    延迟       成本/query
─────────────────────────   ─────   ─────   ───────    ──────────
Naive(向量 Top-3)            62%     —      400ms      $0.005
+多路召回(Hybrid + Top-5)    74%    +12%    450ms      $0.005
+Reranker(Top-50→Top-5)     84%    +10%    700ms      $0.007
+Query 改写(HyDE)            92%    +8%     1200ms     $0.008

总效果:                      +30%    400ms→1200ms      +60%
```

⚠️ **数据局限说明**:上表是业界常见优化模式的归纳数据。实际项目中,三板斧的具体收益受**领域复杂度、文档质量、query 风格**影响显著。某些场景下:

* 多路召回收益可能只有 +3% 到 +20%
* Reranker 在原始问法上可能反而降低准确率(详见第 6 章 6.7.4)
* Query 改写在内部知识库场景上收益较低

**唯一可靠的方法是在自己数据上 A/B 测试**。

***

#### 7.8 三板斧之外的"边际改进" <a href="#id-78-san-ban-fu-zhi-wai-de-andquot-bian-ji-gai-jin-andquot" id="id-78-san-ban-fu-zhi-wai-de-andquot-bian-ji-gai-jin-andquot"></a>

把三板斧用完后,准确率通常能到 85%+。要再往上推到 95%+,需要**针对性优化**——这部分 ROI 通常会快速递减。

🎨 图 7-2:边际改进的几个方向

```
基础三板斧:    62% → 92%   ROI 极高
↓
父子分块:      +1% ~ +3%   工程改造
↓
评估驱动:      +2% ~ +5%   找 bad case → 针对性优化
↓
微调 embedding:+2% ~ +4%   有标注数据时
↓
Agentic RAG:   +3% ~ +5%   复杂场景才值得
↓
微调 LLM:      +1% ~ +3%   投入产出比通常不够
```

**工程上的建议**:把三板斧做扎实后,**不要急着上更高级的技巧**。先做好评估、监控、bad case 分析(第 8、9 章),让数据告诉你下一步该优化什么。

***

#### 7.9 一个常见的"过度优化"反例 <a href="#id-79-yi-ge-chang-jian-de-andquot-guo-du-you-hua-andquot-fan-li" id="id-79-yi-ge-chang-jian-de-andquot-guo-du-you-hua-andquot-fan-li"></a>

📖 某团队的 RAG 上了三板斧后,准确率 86%。team lead 觉得"还可以更好",一个月内陆续加了:

```
+ 上下文压缩(LLMLingua)
+ Query 分解
+ 自我反思(LLM 校验答案)
+ 多 Agent 协同
+ 自定义 Reranker(微调过)
```

结果:

* 准确率从 86% 涨到 88%(+2%)
* 延迟从 1.2s 涨到 6.5s(+440%)
* 成本翻 4 倍
* bug 增多(组件变多,故障面变大)
* 团队精力被分散

教训:**ROI 递减时,应该转向其他优化方向**(数据质量、prompt 工程、用户反馈循环),而不是堆砌技巧。

***

#### 7.10 本章小测验 <a href="#id-710-ben-zhang-xiao-ce-yan" id="id-710-ben-zhang-xiao-ce-yan"></a>

不看答案先想。

1. 三板斧分别解决 RAG 哪一环节的偏差?为什么它们的收益可以叠加?
2. 多路召回的 RRF 中,bm25\_k 和 vector\_k 设多大?为什么不能等于 top\_k?
3. Reranker 为什么不能做"全库召回"?它的工程定位是什么?
4. Query 改写的三种方式各自适合什么场景?
5. 三板斧的应用顺序应该是怎样的?为什么?
6. 三板斧之后的"边际改进"通常 ROI 怎样?给出 2 个常见反例。
7. 一个新接手的 RAG 项目,准确率 60%,你会先做哪一板斧?为什么?
8. 三板斧叠加后延迟通常增加多少?成本通常增加多少?
9. Hybrid 检索时,RRF 用 rank 而不是 score,为什么?
10. 你的 RAG 上了三板斧准确率 85%,继续优化的优先方向应该是什么?

<details>

<summary>👉 参考答案</summary>

1. 板斧 1(Query 改写)解决 query 端的偏差;板斧 2(多路召回)解决检索端的偏差;板斧 3(Reranker)解决排序端的偏差。三个偏差独立,所以收益可以叠加。
2. 通常 50-100。如果等于 top\_k(比如都是 5),Top-5 中 BM25 和 Vector 重合度高,RRF 几乎没融合到不同的结果,效果差。各路召回的 K 必须比最终输出大几倍,RRF 才有融合空间。
3. Reranker 是 Cross-encoder,每个 (query, doc) 对都要过一次 BERT。全库召回意味着对 1 万个 chunks 跑 1 万次推理,单次 query 延迟可能 30 秒。Reranker 的工程定位是"精排",不是"召回",候选数量应该控制在 50-100。
4. 同义词扩展:简单领域、改写规则可枚举;HyDE:用户 query 简短、文档详细,通用场景的起步选择;Multi-Query:对 Recall 极敏感的场景,但延迟成本最高。
5. Query 改写 → 多路召回 → Reranker。理由:Query 改写在 query 阶段就发挥作用,影响后续所有步骤;多路召回扩大候选池;Reranker 在精排阶段最后把关。这个顺序是各板斧依赖关系决定的。
6. 通常快速递减:从 86% 到 88% 可能要 4x 延迟和成本。常见反例:(1)叠加上下文压缩、Query 分解、自我反思、多 Agent、微调 Reranker 等,准确率涨 2% 但延迟涨 440%;(2)花一个月微调 embedding,准确率涨 3%,但同期 prompt 优化能涨 5%。
7. 多路召回(板斧 2)。理由:ROI 最高,改动小(再跑一次 BM25 + RRF 融合),收益稳定(几乎所有场景都有提升),代价可控(BM25 几乎免费)。
8. 延迟通常增加 2-4 倍(400ms → 1200ms 是典型);成本通常增加 50-100%(主要来自 Query 改写的额外 LLM 调用 + Reranker 的推理)。
9. 因为不同检索器的分数不可比——向量是 0-1,BM25 是任意正数。直接相加会让 BM25 的分数主导融合结果。用 rank(1, 2, 3...)统一了量纲,RRF 公式 1/(k+rank) 给排名靠前的项更高权重,排名靠后的衰减。
10. 三板斧叠完后,不要继续堆技巧。优先方向:(1)做好评估体系(第 9 章),用数据驱动找 bad case;(2)做好可观测性(第 8 章),从用户反馈学习;(3)针对 bad case 做精准优化(可能是 prompt、可能是切分、可能是元数据)。盲目堆技巧通常会 ROI 递减且引入复杂度。

</details>

***

#### 7.11 本章总结 <a href="#id-711-ben-zhang-zong-jie" id="id-711-ben-zhang-zong-jie"></a>

<figure><img src=".gitbook/assets/总结 (3).jpg" alt=""><figcaption></figcaption></figure>
