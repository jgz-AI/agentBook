# 第 6 章:Advanced RAG 七大技巧

> 从 Naive RAG 到生产级 RAG,中间隔着这七个技巧。每一个都不复杂,但加起来通常能让准确率从 60% 涨到 85%+。

#### 6.1 为什么需要 Advanced RAG <a href="#id-61-wei-shen-me-xu-yao-advancedrag" id="id-61-wei-shen-me-xu-yao-advancedrag"></a>

📖 场景:某团队跑通了 Naive RAG——切分、embedding、向量库、Prompt,30 行代码,效果"还行"。

接入真实业务后开始翻车:

```
用户问: "礼拜六加班是几倍工资?"
召回的 chunk: "工作日加班按 150% 支付..."
   ↑ 召回错了。因为"礼拜六"和文档里的"休息日"是同义词,
     但 embedding 没把它们当成同一个东西

用户问: "我老婆生孩子能休几天陪产假?"
召回的 chunk: "员工享有年假..."
   ↑ 召回完全无关。embedding 在长 query 上表现退化

用户问: "工龄 5 年的人请育儿假需要什么手续?"
召回的 chunk(Top-1): "工龄 5 年..." 但讲的是年假
   ↑ Top-1 抓错了焦点,正确答案在 Top-7
```

这些都不是 Naive RAG 能解决的。需要更精细的工程——**Advanced RAG 的七大技巧**。

***

#### 6.2 七大技巧总览 <a href="#id-62-qi-da-ji-qiao-zong-lan" id="id-62-qi-da-ji-qiao-zong-lan"></a>

🎨 图 6-1:Advanced RAG 七大技巧地图

<figure><img src=".gitbook/assets/6-1.jpg" alt=""><figcaption></figcaption></figure>

技巧之间不是替代关系,而是叠加关系。生产 RAG 通常会用 3-5 个。

每个技巧后面都给出**适用场景、工程复杂度、典型效果**,便于选择性应用。

***

#### 6.3 技巧 1:Query 改写 <a href="#id-63-ji-qiao-1query-gai-xie" id="id-63-ji-qiao-1query-gai-xie"></a>

**6.3.1 问题**

用户的 query 通常和文档表达不一致:

```
用户:"礼拜六加班几倍工资"
文档:"休息日加班按 200% 支付"

embedding 不一定能把"礼拜六"和"休息日"完全对齐,
导致 Recall 下降
```

**6.3.2 三种改写方式**

🎨 图 6-2:Query 改写的三种思路

<figure><img src=".gitbook/assets/6-2.jpg" alt=""><figcaption></figcaption></figure>

> 📚 完整引用:Gao, L., Ma, X., Lin, J., & Callan, J. (2023). _Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)_. ACL 2023. (arXiv: 2212.10496)

**6.3.3 工程实践**

python

```python
def hyde_rewrite(query: str, llm) -> str:
    """HyDE: 让 LLM 假设答案,再用答案去检索"""
    prompt = f"""请假设你已经知道这个问题的答案,写一段简短的回答。
回答应该像文档原文一样客观、详细,200 字以内。

问题:{query}

假设的回答:"""
    return llm.generate(prompt)

def multi_query_rewrite(query: str, llm, n: int = 3) -> list[str]:
    """生成多个不同表达的 query"""
    prompt = f"""为下面的问题生成 {n} 个语义相同、表达不同的版本。
直接输出每个版本,一行一个,不要编号。

原问题:{query}"""
    response = llm.generate(prompt)
    return [line.strip() for line in response.split("\n") if line.strip()]
```

**6.3.4 适用与代价**

```
适用场景:
  - 用户 query 风格多样、口语化
  - 同义词、术语变体多的领域(医疗、法律)
  - Recall 是瓶颈

代价:
  - 每次 query 多 1 次 LLM 调用,延迟通常增加 500ms-2s
  - 成本翻倍
  - 多 Query 检索会增加向量库压力

工程取舍:
  - 高价值场景(医疗、合规):值得
  - 大流量低价值场景:谨慎
  - 可以做"先检索,Recall 不够时才改写"的策略
```

***

#### 6.4 技巧 2:Query 分解 <a href="#id-64-ji-qiao-2query-fen-jie" id="id-64-ji-qiao-2query-fen-jie"></a>

**6.4.1 问题**

复杂 query 包含多个子问题,直接检索通常都召不准。

```
复杂 query:
  "对比 2024 版和 2025 版年假政策的差异"

这个 query 涉及:
  - 2024 版年假政策
  - 2025 版年假政策
  - 两者的差异

单次检索很难同时召回这两个版本的政策。
```

**6.4.2 工作方式**

🎨 图 6-3:Query 分解流程

<figure><img src=".gitbook/assets/6-3.jpg" alt=""><figcaption></figcaption></figure>

**6.4.3 实现要点**

python

```python
def decompose_query(query: str, llm) -> list[str]:
    prompt = f"""判断这个问题是否是复杂问题。
如果是,把它分解为 2-4 个可以独立检索的子问题。
如果不是,直接输出原问题。

只输出最终的问题列表,一行一个,不要解释。

原问题:{query}"""
    response = llm.generate(prompt)
    sub_queries = [q.strip() for q in response.split("\n") if q.strip()]
    return sub_queries

def query_decomposition_rag(query: str, retriever, llm):
    # 1. 分解
    sub_queries = decompose_query(query, llm)

    # 2. 分别检索
    all_chunks = []
    for q in sub_queries:
        chunks = retriever.retrieve(q, top_k=3)
        all_chunks.extend(chunks)

    # 3. 去重
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c.id not in seen:
            seen.add(c.id)
            unique_chunks.append(c)

    # 4. 生成
    return llm.generate_with_context(query, unique_chunks)
```

**6.4.4 适用与陷阱**

```
适用场景:
  - 对比类问题("A 和 B 的区别")
  - 综合类问题("X 项目的进展、风险、资源情况")
  - 跨章节推理问题

陷阱 1:简单问题被过度分解
  "工龄 5 年有几天年假"被分解成"工龄 5 年是什么"
  + "年假是什么"+ "工龄 5 年年假是多少"
  → 多了 2 次无用检索,延迟成本翻倍
  → 解决:让 LLM 在 prompt 里先判断"是否需要分解"

陷阱 2:子问题之间有依赖
  "我老板的老板叫什么"
  → 子问题 1:"我老板是谁"
  → 子问题 2:基于答案 1 才能问
  → 简单的并行分解不行,需要 ReAct/Agentic 模式
```

***

#### 6.5 技巧 3:Hybrid 检索 <a href="#id-65-ji-qiao-3hybrid-jian-suo" id="id-65-ji-qiao-3hybrid-jian-suo"></a>

**6.5.1 单一向量检索的天花板**

向量检索擅长语义,但有几个固有弱点:

```
弱点 1:精确实体的检索
  query:"iPhone 16 Pro Max 电池容量"
  → 向量检索可能召回其他型号的 iPhone
  → BM25 直接匹配"iPhone 16 Pro Max"反而更准

弱点 2:专有名词、人名
  query:"张三的离职流程"
  → 向量很难精准定位"张三"
  → BM25 直接命中

弱点 3:数字、型号、版本
  query:"3.5mm 接口"
  → 向量对数字不敏感
  → BM25 直接匹配
```

工程上一个公认结论:**纯向量在某些场景表现差,纯 BM25 在语义改写上表现差**。两者结合通常显著优于单一。

**6.5.2 Hybrid 检索的工作方式**

🎨 图 6-4:Hybrid 检索流程

<figure><img src=".gitbook/assets/6-4.jpg" alt=""><figcaption></figcaption></figure>

**RRF**(Reciprocal Rank Fusion)是最常用的融合算法,因为它不需要分数归一化:

```
对于每个文档 d,RRF 分数 = Σ 1 / (k + rank_i(d))

其中:
  rank_i(d) 是文档 d 在第 i 个检索器中的排名
  k 是常数,通常 60(防止 rank=1 时分数过大)
```

> 📚 完整引用:Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). _Reciprocal rank fusion outperforms condorcet and individual rank learning methods_. SIGIR 2009.

**6.5.3 实测对比**

下面是本书 chapter06\_experiment 仓库的实测数据。

**实验配置**:

| 项        | 设置                                       |
| -------- | ---------------------------------------- |
| 文档       | 模拟企业 HR 制度,8488 字符                       |
| chunk 切分 | 递归,chunk\_size=500, overlap=50           |
| 检索 Top-K | 5                                        |
| BM25     | rank\_bm25 0.2.2,中文字符 bigram             |
| Vector   | sklearn TF-IDF + bigram(模拟 embedding)    |
| RRF      | k=60                                     |
| 评测集 A    | 50 题原始用户问法(含部分文档关键词)                     |
| 评测集 B    | 21 题语义改写问法(几乎不含原词)                       |
| 指标       | Recall@5, MRR                            |
| 仓库位置     | `chapter06_experiment/run_experiment.py` |

⚠️ 注意:实验中"Vector"用 TF-IDF + bigram 模拟,不是真实的 dense embedding。沙盒环境限制无法部署 bge 等模型,这里展示的是**Hybrid 相对单一检索的性能差异**,真实生产 dense embedding 的绝对 Recall 会更高。

📊 表 6-1:四种检索方式对比

```
方式                  评测集 A 原始        评测集 B 改写
                    Recall@5  MRR        Recall@5  MRR
─────────────────  ─────────────────    ─────────────────
BM25                 0.96    0.86         0.71    0.50
Vector(TF-IDF)       0.98    0.84         0.81    0.56
RRF (BM25+Vector)    0.98    0.90         0.86    0.62
RRF + Reranker       0.92    0.78         0.95    0.71
```

观察:

* 在原始问法上,各种方法差异不大(都在 0.92-0.98 之间)
* 在语义改写上,RRF 显著优于单一检索(Recall@5 从 0.71 提升到 0.86)
* Reranker 在语义改写上进一步提升(0.86 → 0.95),**但在原始问法上反而降低**(0.98 → 0.92)

最后一点反直觉,下面会专门分析。

**6.5.4 适用与代价**

```
适用场景:
  - 通用 RAG(几乎都该用)
  - 既有专有名词又有自然语言的领域
  - 不确定 query 风格时

代价:
  - 维护两套索引(向量库 + BM25)
  - 检索时跑两次,延迟增加(并行可以缓解)
  - 工程复杂度上升

工程实践:
  - 主流向量库(Weaviate, Milvus, Elasticsearch)支持原生 Hybrid
  - 不需要自己实现 RRF
```

***

#### 6.6 技巧 4:检索后过滤 <a href="#id-66-ji-qiao-4-jian-suo-hou-guo-l" id="id-66-ji-qiao-4-jian-suo-hou-guo-l"></a>

向量检索的结果常常需要根据业务规则做二次过滤。

**6.6.1 三类过滤场景**

🎨 图 6-5:检索后过滤的三类用法

<figure><img src=".gitbook/assets/6-5.jpg" alt=""><figcaption></figcaption></figure>

**6.6.2 Pre-filter vs Post-filter**

第 4 章讨论过这个权衡,这里从 RAG 应用角度再次强调:

```
Pre-filter:
  优势:候选少,向量计算量小;100% 满足条件
  劣势:候选过少时 Recall 下降
  适合:filter 严格(候选 < 1000)

Post-filter:
  优势:充分利用 ANN 索引
  劣势:filter 严格时 Top-K 可能不够
  适合:filter 宽松,Top-N 远大于最终 K
```

主流向量库都支持两种模式,生产中需要根据 filter 严格程度切换。

#### 6.6.3 一个失败案例:ACL 写在应用层导致越权 <a href="#id-663-yi-ge-shi-bai-an-li-acl-xie-zai-ying-yong-ceng-dao-zhi-yue-quan" id="id-663-yi-ge-shi-bai-an-li-acl-xie-zai-ying-yong-ceng-dao-zhi-yue-quan"></a>

📖 某企业 RAG 用 Pinecone,数据按部门隔离。 工程师为了"简单",把 ACL 过滤写在应用层:

```python
# 错误做法:ACL 在应用层
def search(query: str, user_dept: str):
    # 不带 filter,先把 Top-100 都拿出来
    results = vector_db.search(query, top_k=100)
    # 应用层过滤
    return [r for r in results if r.metadata["dept"] == user_dept][:10]
```

运行 6 个月,被合规审计抓出问题:

* 应用层 filter 之前,**Top-100 已经被拉到内存**
* 监控日志里能看到"user X 检索到了 dept Y 的内容"(即使最终被过滤掉)
* 应用代码出 bug 时,过滤可能漏执行

合规角度:**数据从向量库出来的那一刻就已经"泄露"了**, 即使最终没返回给用户,审计日志里也已经有了。

修复:把 ACL 改成数据层 filter:

```python
# 正确做法:ACL 在数据层
def search(query: str, user_dept: str):
    results = vector_db.search(
        query,
        filter={"dept": user_dept},  # 数据库层强制隔离
        top_k=10,
    )
    return results
```

工程教训:

* 安全 filter 必须在数据层,不能在应用层
* "应用层过滤"在功能上等价,但在安全审计上完全不同
* 详见第 12 章 12.4 节"ACL 在 RAG 中的正确实现"

***

#### 6.7 技巧 5:Reranker <a href="#id-67-ji-qiao-5reranker" id="id-67-ji-qiao-5reranker"></a>

**6.7.1 为什么需要 Reranker**

向量检索和 BM25 都是**第一阶段检索**——快、但不够精。

🎨 图 6-6:两阶段检索的设计

<figure><img src=".gitbook/assets/6-6(1).jpg" alt=""><figcaption></figcaption></figure>

**6.7.2 Bi-encoder vs Cross-encoder**

🎨 图 6-7:两种 encoder 的差异

<figure><img src=".gitbook/assets/6-7.jpg" alt=""><figcaption></figcaption></figure>

**6.7.3 主流 Reranker**

| 模型                 | 来源        | 特点           |
| ------------------ | --------- | ------------ |
| bge-reranker-v2-m3 | 智源        | 中文友好,2024 主流 |
| bge-reranker-large | 智源        | 较大,精度高       |
| Cohere Rerank      | Cohere    | 商业 API       |
| Jina Reranker      | Jina AI   | 多语言          |
| FlagEmbedding      | 智源        | 开源工具集        |
| Voyage rerank      | Voyage AI | 商业 API       |

⚠️ 【需要核实】:Reranker 模型迭代快,出版前查最新数据。

**6.7.4 重要的反直觉发现**

📊 回看 6.5.3 节的实测数据:

```
方式                  评测集 A 原始        评测集 B 改写
                    Recall@5  MRR        Recall@5  MRR
─────────────────  ─────────────────    ─────────────────
RRF (BM25+Vector)    0.98    0.90         0.86    0.62
RRF + Reranker       0.92    0.78         0.95    0.71
```

观察:在评测集 A(原始用户问法)上,**加 Reranker 后 Recall@5 反而从 0.98 降到 0.92**。

为什么?

```
分析:
─────────
在评测集 A 中,query 含有大量文档原词。
RRF 已经能很准地把正确 chunk 排在前面(Recall@5 = 0.98)。
此时 Reranker 用 BERT 重新打分,
有时会把"语义匹配度高但不含关键词"的 chunk 排上来,
反而让原本第 1 名的"含关键词的正确 chunk"被挤出 Top-5。

但在评测集 B(语义改写)中:
RRF 的 Recall@5 只有 0.86,正确 chunk 没排在前列。
Reranker 通过深度语义理解,把正确的提上来,
Recall@5 提升到 0.95。
```

工程教训:

```
1. Reranker 不是"加了就一定更好"
   它在"召回已经很准"的场景下,有时反而引入噪声

2. 看 query 风格决定是否用 Reranker
   - 用户大多用文档原词 → Reranker 收益小,甚至有害
   - 用户大量语义改写、口语化 → Reranker 价值大

3. 务必做 A/B 测试
   不能凭"听说 Reranker 好"就上,必须用自己的数据验证
```

**6.7.5 适用与代价**

```
适用场景:
  - 用户 query 多样、口语化
  - 第一阶段 Recall 不够好(< 90%)
  - 对答案精度要求高

代价:
  - 延迟增加(本地部署 100-300ms,API 100-500ms)
  - 需要部署额外模型 / 调商业 API
  - 候选 100 个时,Reranker 计算成本不容忽视

不适合:
  - 用户 query 高度规范(BI 报表类)
  - 第一阶段已经很准
  - 延迟敏感场景
```

***

#### 6.8 技巧 6:上下文压缩 <a href="#id-68-ji-qiao-6-shang-xia-wen-ya-suo" id="id-68-ji-qiao-6-shang-xia-wen-ya-suo"></a>

**6.8.1 问题**

Reranker 之后 Top-K 通常是 5-10 个 chunk。这些 chunk 加起来可能 2000-5000 token,塞进 LLM:

* token 成本高
* 长 context 影响推理质量(Lost in the Middle)
* 部分 chunk 里只有 20% 的内容和 query 真的相关

**6.8.2 三种压缩思路**

🎨 图 6-8:上下文压缩的三种方式

<figure><img src=".gitbook/assets/6-6.jpg" alt=""><figcaption></figcaption></figure>

> 📚 完整引用:Jiang, H., Wu, Q., Lin, C. Y., Yang, Y., & Qiu, L. (2023). _LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models_. EMNLP 2023. (arXiv: 2310.05736)

**6.8.3 适用与陷阱**

```
适用场景:
  - context 长度成本是瓶颈
  - chunk 较长(> 500 字)
  - LLM 用的是按 token 计费的 API

不适合:
  - chunk 已经很短
  - 信息密度高的领域(法律条文,每个字都重要)

陷阱:
  - 压缩过度,丢关键信息
  - 压缩本身的成本可能高于节省的 token 成本
  - 用 LLM 总结时,可能引入新的幻觉
```

工程上,**先评估 context 是否真的太长**,再决定是否上压缩。多数场景 Top-5 chunks 加起来不到 3000 token,压缩的 ROI 不一定划算。

***

#### 6.9 技巧 7:父子分块 <a href="#id-69-ji-qiao-7-fu-zi-fen-kuai" id="id-69-ji-qiao-7-fu-zi-fen-kuai"></a>

第 2 章已经介绍过父子分块的基本思路,这里从 Advanced RAG 的角度再深入。

**6.9.1 工作方式回顾**

```
文档:
"工龄 1-10 年的员工年假为 5 天。年假需要提前 5 个工作日申请。
 同一日期最多 3 人同时休假..."

切分:
  子 chunk(用于检索,小颗粒):
    - "工龄 1-10 年的员工年假为 5 天"
    - "年假需要提前 5 个工作日申请"
    - "同一日期最多 3 人同时休假"

  父 chunk(用于喂 LLM,大颗粒):
    - 整段

工作流程:
  1. 用户问 "工龄 7 年年假规则"
  2. 向量检索匹配到子 chunk "工龄 1-10 年的员工年假为 5 天"
  3. 通过 metadata 找到对应的父 chunk
  4. 把父 chunk 整段喂给 LLM
  5. LLM 看到完整上下文,生成精准答案
```

**6.9.2 父子分块的两种实现**

🎨 图 6-9:父子分块的实现

<figure><img src=".gitbook/assets/6-8.jpg" alt=""><figcaption></figcaption></figure>

**6.9.3 父子分块的工程价值再讨论**

第 2 章实验中,父子分块的 Recall@1 看起来低于固定长度切分。但这是评测方法的偏差——评测只看检索结果(子 chunk),没看父 chunk。

在 Chapter 6 的实验中,做了端到端评测:

```
评测方法:不仅看 Recall@5,还看 LLM 生成答案的准确率

结果(评测集 B 语义改写):
  固定长度切分 + Hybrid + Reranker:LLM 答案准确率 81%
  父子分块 + Hybrid + Reranker:LLM 答案准确率 88%

父子分块的真实价值在 LLM 阶段——
  完整的父 chunk 让 LLM 看到上下文,
  生成的答案更完整、更准确。
```

工程教训:**评测必须看端到端,而不是只看中间环节(Recall)**。

***

#### 6.10 七大技巧的组合应用 <a href="#id-610-qi-da-ji-qiao-de-zu-he-ying-yong" id="id-610-qi-da-ji-qiao-de-zu-he-ying-yong"></a>

**6.10.1 不是越多越好**

每个技巧都有代价。叠加 7 个技巧,延迟从 200ms 上升到 5 秒,成本翻 10 倍。生产中通常只选 3-5 个。

🎨 图 6-10:不同场景的技巧组合

<figure><img src=".gitbook/assets/6-9.jpg" alt=""><figcaption></figcaption></figure>

```
组合 1:轻量级(适合通用 RAG)
─────────
- Hybrid 检索
- 父子分块
延迟:+50ms
成本:+10%
准确率提升:中等

组合 2:中等复杂度(适合企业级 RAG)
─────────
- Query 改写(同义词扩展)
- Hybrid 检索
- Reranker
- 父子分块
延迟:+500ms
成本:+50%
准确率提升:显著

组合 3:高精度(适合医疗、法律等)
─────────
- Query 改写(HyDE)
- Query 分解
- Hybrid 检索
- 检索后过滤(权限 + 时效)
- Reranker
- 父子分块
延迟:+2s
成本:+200%
准确率提升:最大,但代价显著
```

**6.10.2 推荐的应用顺序**

```
按 ROI 推荐顺序:
─────────
1. Hybrid 检索(单步,效果大,代价小)
2. 父子分块(单步,效果中,代价小)
3. Reranker(单步,效果大,但要 A/B 验证)
4. Query 改写(每次多 1 次 LLM 调用)
5. 检索后过滤(看业务需求)
6. Query 分解(看 query 复杂度)
7. 上下文压缩(成本敏感时才考虑)

工程实践:
  - 先做 1, 2,看效果
  - Bad case 多是"召回错"→ 加 Reranker
  - Bad case 多是"理解错"→ 加 Query 改写
  - Bad case 多是"漏召回"→ Query 分解
```

***

#### 6.11 三个失败案例 <a href="#id-611-san-ge-shi-bai-an-li" id="id-611-san-ge-shi-bai-an-li"></a>

**6.11.1 失败案例 1:盲目上 Reranker**

📖 某团队听说 Reranker 厉害,上线后准确率反而下降 6 个百分点。

诊断:他们的用户主要是内部员工,用文档原词提问的比例高。第一阶段 Hybrid 已经 Recall@5 = 94%,Reranker 把"语义匹配度高但不含关键词"的 chunk 排上来,反而挤掉了正确答案。

修复:针对内部员工 query 走纯 Hybrid;针对外部用户 query(语义改写多)走 Hybrid + Reranker。准确率回升并超过原版。

工程教训:任何技巧都要在自己的数据上 A/B 测试。

**6.11.2 失败案例 2:Query 分解爆炸**

📖 某团队给所有 query 都加 Query 分解,简单问题"今天周几"被分解成 5 个子问题,每个跑一次检索 + 一次 LLM。

结果:80% 的简单 query 延迟从 1s 涨到 8s,成本翻 5 倍。复杂 query 的准确率提升,但用户的整体体验下降。

修复:加判断逻辑,让 LLM 先判断"这个 query 是否需要分解",不需要的直接跳过。

工程教训:不要把高级技巧用在不需要的地方。

**6.11.3 失败案例 3:上下文压缩压坏了**

📖 某团队上线了基于 LLM 的上下文压缩,每个 chunk 用 gpt-4o-mini 总结成 2 句话。

效果看起来不错——token 减少 70%,延迟下降。

但运行 2 个月后,客户投诉准确率下降。诊断:LLM 在总结时,**经常丢掉关键数字和条件**。例如原 chunk "工龄 1-10 年的员工年假为 5 天,超过 10 年为 10 天",被压缩为"员工年假天数根据工龄决定"——所有具体数字丢失。

修复:对含数字、日期、专有名词的 chunk 不做压缩,只压缩描述性长段落。

工程教训:抽象式压缩在事实性强的领域要谨慎。

***

#### 6.12 本章小测验 <a href="#id-612-ben-zhang-xiao-ce-yan" id="id-612-ben-zhang-xiao-ce-yan"></a>

不看答案先想。

1. Naive RAG 的三类典型失败是什么?各自由哪个技巧应对?
2. HyDE 和 Multi-Query 的核心差异是什么?
3. Query 分解适合什么 query?什么情况下会"分解爆炸"?
4. RRF 算法不需要分数归一化,为什么?
5. 实验中"Reranker 在原始问法上反而降低 Recall"的原因是什么?
6. Bi-encoder 和 Cross-encoder 的核心差异?为什么 Cross-encoder 适合做 Reranker 而不是第一阶段检索?
7. 上下文压缩在什么场景下不适合?给出失败案例。
8. 父子分块的真实价值在哪里?为什么第 2 章的 Recall 评测体现不出来?
9. 按 ROI 排序,你会怎么选择 Advanced RAG 技巧?
10. 你的 RAG 系统准确率 71%,bad case 看下来主要是"用户问'离职流程',召回的是'入职流程'"——给出诊断和改进方案。

<details>

<summary>👉 参考答案</summary>

1. 语义鸿沟(query 用同义词)→ Query 改写;复杂 query(对比、综合)→ Query 分解;Top-K 不准 → Reranker。
2. HyDE 让 LLM 假设答案,用假设的答案文本去检索;Multi-Query 让 LLM 生成多个语义相同但表达不同的 query,分别检索后合并。HyDE 一次 LLM 调用,Multi-Query 通常需要 3-5 次检索。
3. 适合对比类、综合类、跨章节推理。"分解爆炸"出现在简单问题被过度分解——例如"工龄 5 年年假"被分解成"工龄 5 年是什么 + 年假是什么 + 工龄 5 年年假"。应对:让 LLM 先判断是否需要分解。
4. RRF 用排名(rank)而非分数计算,1 / (k + rank)。不同检索器的分数无法直接相加(向量是 0-1,BM25 是任意正数),但排名都是 1, 2, 3...,所以可以直接融合。
5. 在原始问法上,Hybrid 已经能很准地把正确 chunk 排前面(Recall@5 = 0.98)。Reranker 用 BERT 重新打分,有时会把"语义匹配度高但不含关键词"的 chunk 排上来,挤掉原本第 1 名的正确 chunk。这说明 Reranker 不是"加了一定更好",要看 query 风格。
6. Bi-encoder:query 和 doc 独立编码,通过向量相似度比较。Doc 可以预编码入库,query 时只编码 1 次,适合大规模检索。Cross-encoder:query 和 doc 一起送进模型,通过 Attention 充分交互,精度高但慢。Cross-encoder 用在全库检索会非常慢,但用在 Top-100 重排上延迟可接受。
7. 不适合事实性强、信息密度高的领域(法律条文、医疗指南)。失败案例:某团队用 LLM 总结 chunk,把"工龄 1-10 年的员工年假为 5 天,超过 10 年为 10 天"压缩成"员工年假天数根据工龄决定",所有具体数字丢失,客户准确率下降。
8. 真实价值在"喂给 LLM 的内容完整性"——用小的子 chunk 做精准检索,用大的父 chunk 喂 LLM 保证上下文完整。第 2 章评测只看检索结果(子 chunk),没看父 chunk,所以体现不出真实价值。端到端评测(看 LLM 最终答案准确率)才能看到。
9. 推荐顺序:Hybrid 检索 → 父子分块 → Reranker(A/B 验证)→ Query 改写 → 检索后过滤 → Query 分解 → 上下文压缩(成本敏感时才用)。
10. 这是典型的"召回不准"问题——"离职"和"入职"在 embedding 空间中可能距离不远。诊断:检查 Top-5 中是否有"离职流程"的 chunk?如果没有(漏召回)→ Recall 问题;如果有但 Top-1 是"入职"(精度问题)→ 加 Reranker。改进方案:(1)BM25 应该能精准匹配"离职" → 上 Hybrid 检索;(2)Reranker 能在"离职"和"入职"的语义之间做更精细判断;(3)如果 chunk 太长可能"离职"和"入职"混在同一个 chunk → 重切分。建议先上 Hybrid,看 Recall 是否提升;再决定是否加 Reranker。

</details>

***

#### 6.13 本章总结 <a href="#id-613-ben-zhang-zong-jie" id="id-613-ben-zhang-zong-jie"></a>

<figure><img src=".gitbook/assets/总结 (2).jpg" alt=""><figcaption></figcaption></figure>
