# 第 5 章:Prompt 工程

> 同样的 Top-K chunks,不同的 prompt 让 LLM 答出 60% 准确率或 92% 准确率——这是 Prompt 工程的真实影响。

#### 5.1 一个被低估的 ROI <a href="#id-51-yi-ge-bei-di-gu-de-roi" id="id-51-yi-ge-bei-di-gu-de-roi"></a>

📖 场景:某团队的 RAG 系统准确率卡在 71%。team lead 想升级 LLM、加 Reranker,预算 5000 美元/月。

实习生用了 2 天改了 prompt 模板,准确率涨到 84%——0 美元成本。

工程上一个常见误判:大家把 Prompt 工程当"花拳绣腿",疯狂砸资源去优化别的环节。实际上**Prompt 工程在 RAG 系统中的 ROI 通常是最高的——零成本、高回报、快速验证**。

这一章给一套**可复用的 RAG Prompt 设计方法**,不教你写诗,只教你写能让 LLM "听话"的指令。

***

#### 5.2 RAG Prompt 的核心结构 <a href="#id-52ragprompt-de-he-xin-jie-gou" id="id-52ragprompt-de-he-xin-jie-gou"></a>

🎨 图 5-1:RAG Prompt 的 6 层结构

<figure><img src=".gitbook/assets/5-1.jpg" alt=""><figcaption></figcaption></figure>

这 6 层不是必须全用,但生产级 RAG 通常至少包含 Layer 1-4。

***

#### 5.3 Layer 1-3:System Prompt 怎么写 <a href="#id-53layer13systemprompt-zen-me-xie" id="id-53layer13systemprompt-zen-me-xie"></a>

**5.3.1 角色定位**

🎨 图 5-2:三种角色定位的对比

<figure><img src=".gitbook/assets/5-2.jpg" alt=""><figcaption></figcaption></figure>

实测中,版本 C 比版本 A 的答案质量通常显著高。原因:角色定位让 LLM 进入特定的"语境模式",回答更稳定。

但**不要无限堆砌角色描述**。过长的角色 prompt 会:

* 占用 context 预算
* 让 LLM 抓不住重点
* 引入歧义

推荐做法:角色定位控制在 100-300 字以内,只描述与任务直接相关的信息。

**5.3.2 行为边界**

行为边界是 RAG Prompt 最重要的部分,直接影响幻觉率。

防幻觉的边界 prompt

```
基础版(常用):
─────────
"只使用上述资料回答问题。
 如果资料中没有相关信息,请回答'根据提供的资料,我无法回答这个问题'。
 不要凭借常识或其他知识补充。"

进阶版(高精度场景):
─────────
"严格按照以下规则:
 1. 答案必须从资料中直接找到依据
 2. 不要做任何推理或推断
 3. 不要补充资料外的信息(即使是常识)
 4. 如果资料中信息矛盾,指出矛盾点
 5. 如果资料不足以完整回答,只回答能确定的部分
 6. 在 [资料引用] 部分列出你引用的具体段落"
```

实测发现:加入"必须从资料中直接找到依据"这一条,通常能让幻觉率显著下降。

**5.3.3 回答格式**

格式约束的工程价值常被低估。LLM 不约束格式时,输出风格波动大,影响下游处理。

```
常见格式约束:
─────────

控制长度:
  "用 3-5 句话简洁回答"
  "回答不超过 200 字"

控制结构:
  "用 markdown 格式,关键信息加粗"
  "分点回答(数字编号)"

控制元素:
  "在末尾标注引用编号 [1][2]"
  "如果涉及数字,用表格呈现"
  "如果是流程,用步骤列表"

控制语气:
  "用专业但易懂的语气"
  "避免使用'根据资料'这种程式化开头"
```

工程上一个推荐:**生产 RAG 应该用结构化输出**(JSON / 固定 markdown 段落),便于下游解析和监控。

***

#### 5.4 Layer 4:Context 的组织 <a href="#id-54layer4context-de-zu-zhi" id="id-54layer4context-de-zu-zhi"></a>

参考资料(context)怎么组织,对 LLM 的理解影响极大。

**5.4.1 三种 context 组织方式**

🎨 图 5-3:Context 组织方式对比

<figure><img src=".gitbook/assets/5-3.jpg" alt=""><figcaption></figcaption></figure>

**5.4.2 Context 顺序的影响**

LLM 对 context 中不同位置的内容,注意力是不均匀的。

🎨 图 5-4:Lost in the Middle 现象

<figure><img src=".gitbook/assets/5-4.jpg" alt=""><figcaption></figcaption></figure>

> 📚 完整引用:Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). _Lost in the Middle: How Language Models Use Long Contexts_. TACL. (arXiv: 2307.03172)

工程实践:

python

```python
def organize_context(chunks: list, query: str) -> str:
    # chunks 已按相关性降序排列
    # 把最相关的放在最后(最接近 query)
    reordered = list(reversed(chunks))

    parts = []
    for i, chunk in enumerate(reordered, 1):
        parts.append(f"[{i}] {chunk.text}")

    return "参考资料:\n\n" + "\n\n".join(parts)
```

***

#### 5.5 防幻觉的进阶技巧 <a href="#id-55-fang-huan-jue-de-jin-jie-ji-qiao" id="id-55-fang-huan-jue-de-jin-jie-ji-qiao"></a>

**5.5.1 "明示拒答"模板**

很多 LLM 倾向于"凑出答案",即使资料不足。要主动训练它说"我不知道"。

```
普通 Prompt:
─────────
"基于资料回答问题。"
→ LLM 可能在资料不充分时还硬答

强化 Prompt:
─────────
"基于资料回答。判断流程:
 1. 资料中是否直接提到这个问题的答案?
    - 是 → 直接引用回答
    - 否 → 进入步骤 2
 2. 资料中是否有相关但不完整的信息?
    - 是 → 回答'资料中有相关信息但不完整,具体如下:...'
    - 否 → 回答'资料中没有相关信息,建议咨询 HR 部门'

 不要在资料外凭借常识补充。"
```

实测中,这种"判断流程化"的 prompt 通常能让"硬答"显著减少。

**5.5.2 引用要求**

让 LLM 给出引用,有两个工程价值:

```
1. 用户可以核对
   答案错了能快速定位到哪条资料出错

2. 强制 LLM "对齐"资料
   要求引用 → LLM 必须从资料找内容 → 减少幻觉
```

引用的常见格式:

```
方式 1:行内引用
"工龄 5 年有 5 天年假 [1]。年假需要提前 5 个工作日申请 [3]。"

方式 2:末尾汇总
"工龄 5 年有 5 天年假。年假需要提前 5 个工作日申请。

 引用:[1] 员工手册 §3.1  [3] 员工手册 §3.5"

方式 3:JSON 输出
{
  "answer": "工龄 5 年有 5 天年假",
  "citations": [
    {"chunk_id": 1, "source": "员工手册 §3.1"}
  ],
  "confidence": "high"
}
```

生产级 RAG 通常用方式 3(结构化),便于程序处理。

**5.5.3 自我校验**

更进阶的做法:让 LLM 自己校验答案。

python

```python
def two_stage_answer(query: str, context: str) -> dict:
    # 第一阶段:生成答案
    answer = llm.generate(
        prompt=f"参考资料:{context}\n\n问题:{query}\n\n回答:"
    )

    # 第二阶段:校验答案
    check = llm.generate(
        prompt=f"""请校验下面的回答是否完全基于提供的资料。

        资料:{context}
        问题:{query}
        回答:{answer}

        判断:
        - 回答中的每个事实点是否都在资料中?
        - 是否有资料外的信息?
        - 是否有遗漏?

        输出 JSON:
        {{"is_faithful": true/false, "issues": [...]}}
        """
    )

    return {"answer": answer, "check": check}
```

代价:每次回答要调 2 次 LLM,贵且慢。适合高精度场景,如医疗、法律。

***

#### 5.6 Few-shot:用例子教 LLM <a href="#id-56fewshot-yong-li-zi-jiao-llm" id="id-56fewshot-yong-li-zi-jiao-llm"></a>

当 LLM 难以理解任务时,给几个例子通常显著有效。

🎨 图 5-5:Few-shot 在 RAG 中的应用

<figure><img src=".gitbook/assets/5-5.jpg" alt=""><figcaption></figcaption></figure>

工程上:

```
Few-shot 的常见数量:2-5 个例子
太少:模式不清晰
太多:占用 context,边际收益递减

例子的选择:
  - 覆盖典型场景(直接答 / 拒答 / 综合多条资料)
  - 覆盖边界 case(资料矛盾、信息不全)
  - 不要用太长的例子
```

***

#### 5.7 Prompt 的版本管理 <a href="#id-57prompt-de-ban-ben-guan-li" id="id-57prompt-de-ban-ben-guan-li"></a>

生产 RAG 的 prompt 通常会反复改。版本管理常被忽视。

**5.7.1 一个真实失败案例**

📖 某团队的 RAG 系统在线上跑了半年。有一天产品经理来反馈:"为什么准确率最近降了 8 个百分点?"

工程师查代码,发现 prompt 被某次提交改过——某个紧急修复加了"如果用户问敏感问题,委婉拒绝"这一条。这条改动解决了一个具体 case,但让一些**正常问题也被错误拒答**。

问题是:**没人记得是什么时候改的、是谁改的、为什么改的**。

修复后,团队建立了 prompt 版本管理流程:

* prompt 单独存为 yaml / 数据库
* 每次改动有 changelog
* 每次改动跑评测集,记录指标变化
* 上线前 A/B 对照

**5.7.2 工程实践**

python

```python
# prompts/v3.yaml
version: v3
created_at: 2026-03-15
author: alice
description: 加强"拒答"行为,降低幻觉

system: |
  你是 ACME 公司的 HR 助手...

user_template: |
  参考资料:
  {context}

  问题:{query}

  请基于资料回答...

# 评测记录
evaluation:
  date: 2026-03-15
  test_set: hr_v2.jsonl
  metrics:
    accuracy: 0.86
    hallucination_rate: 0.04
    refusal_rate: 0.08
  vs_v2:
    accuracy: +0.03
    hallucination_rate: -0.06
    refusal_rate: +0.05  # 拒答增加,但幻觉显著下降
```

每次改 prompt 都跑一次评测集,记录变化。这是生产级 RAG 的基本工程纪律。

***

#### 5.8 Prompt 工程的常见误区 <a href="#id-58prompt-gong-cheng-de-chang-jian-wu-qu" id="id-58prompt-gong-cheng-de-chang-jian-wu-qu"></a>

🎨 图 5-6:三个常见误区

<figure><img src=".gitbook/assets/5-6.jpg" alt=""><figcaption></figcaption></figure>

***

#### 5.9 中文 RAG 的 Prompt 特殊性 <a href="#id-59-zhong-wen-rag-de-prompt-te-shu-xing" id="id-59-zhong-wen-rag-de-prompt-te-shu-xing"></a>

中英文 LLM 的行为差异,有些是文化的,有些是模型训练数据导致的。

**5.9.1 中文 Prompt 的常见差异**

```
1. 中文 LLM 对"礼貌"敏感
   - 多用"请""谢谢"等敬语,通常有积极效果
   - 但不要过度,变成废话

2. 中文 LLM 对结构化指令的依从性更强
   - 用"步骤 1、步骤 2"
   - 用"判断"+"如果...则..."
   - 用 markdown 标题

3. 中文 LLM 容易过度礼貌
   - "如果资料不足请回答"我不知道""
   - LLM 可能加上"很抱歉给您带来困扰"等冗余
   - 需明确"直接说不知道,不要客套"

4. 中英文混合时
   - 明确指定输出语言
   - 例:"用中文回答,即使资料是英文"
```

**5.9.2 中文 RAG 的 prompt 模板**

yaml

```yaml
system: |
  你是 [公司] 的智能助手。基于提供的公司资料,回答用户问题。

  回答原则:
  1. 严格基于资料,资料中没有的信息不要补充
  2. 资料不足以回答时,直接说"根据现有资料,无法回答这个问题"
  3. 回答简洁,3-5 句话内
  4. 涉及数字时,精确引用资料原文
  5. 在回答末尾标注引用编号 [1][2]

  不要做的事:
  - 不要添加"很抱歉""请理解"等客套话
  - 不要解释"我是 AI"
  - 不要在拒答时给出"建议你..."以外的内容

user_template: |
  参考资料:

  {context}

  问题:{query}

  请基于资料回答:
```

***

#### 5.10 本章小测验 <a href="#id-510-ben-zhang-xiao-ce-yan" id="id-510-ben-zhang-xiao-ce-yan"></a>

不看答案先想。

1. RAG Prompt 的 6 层结构是什么?哪几层是必备的?
2. 为什么要给 LLM 设定"角色定位"?角色描述应该有多长?
3. Lost in the Middle 现象是什么?在工程上怎么应对?
4. "明示拒答"模板和普通 prompt 的核心差异是什么?
5. 强制要求 LLM 给出引用,有什么工程价值?
6. Few-shot 在 RAG 中的常见用法是什么?
7. 为什么 prompt 需要版本管理?给出一个真实失败案例。
8. 中文 RAG 的 prompt 和英文 RAG 有哪些差异?
9. 你的 RAG 系统幻觉率较高,prompt 上有哪些改进方向?
10. 你的 RAG 系统拒答率过高(用户觉得"什么都不知道"),怎么调整 prompt?

<details>

<summary>👉 参考答案</summary>

1. 6 层:角色定位、行为边界、回答格式、参考资料、用户问题、输出引导。其中 Layer 1(角色)、Layer 2(边界)、Layer 4(资料)、Layer 5(问题)通常是必备。Layer 3(格式)和 Layer 6(引导)看场景。
2. 让 LLM 进入特定的"语境模式",回答更稳定、更符合业务调性。描述长度推荐 100-300 字,只描述与任务直接相关的信息,过长反而让 LLM 抓不住重点。
3. LLM 对 context 中开头和结尾的注意力高于中间。工程上:Top-K chunks 按相关性排序后,把最相关的放在最后(最接近 query)。这是 Reranker 价值的来源之一。
4. 普通 prompt 简单说"基于资料回答",LLM 在资料不足时容易"凑答案"。"明示拒答"模板把判断过程流程化("步骤 1:资料中是否直接提到...如果否,进入步骤 2"),让 LLM 主动判断"什么时候应该说不知道"。实测中通常能让硬答显著减少。
5. (1)用户可以核对,答案错了能定位;(2)强制 LLM 对齐资料,要求引用 = LLM 必须从资料找内容 = 减少幻觉。生产 RAG 通常用结构化 JSON 输出,便于程序处理。
6. 当 LLM 难以理解任务时,给 2-5 个例子。常见用法:覆盖典型场景(直接答 / 拒答 / 综合多条资料)+ 边界 case(资料矛盾、信息不全)。
7. Prompt 是 RAG 系统的核心配置,会反复改。没有版本管理时,某次"紧急修复"可能引入回归,但没人记得改了什么。失败案例:某团队加了"敏感问题委婉拒绝"的规则,解决了某个 case,但让正常问题也被错误拒答,准确率掉 8%。生产实践:prompt 单独存 yaml + changelog + 每次改动跑评测。
8. (1)中文 LLM 对"礼貌"敏感,适度敬语有积极效果;(2)中文 LLM 对结构化指令依从性更强(步骤、判断);(3)中文 LLM 容易过度礼貌,需明确"直接说不知道,不客套";(4)中英文混合时明确指定输出语言。
9. (1)加强"明示拒答"——用流程化指令告诉 LLM 何时拒答;(2)强制引用要求,让 LLM 必须从资料找内容;(3)Few-shot 加几个拒答的例子;(4)加自我校验阶段(代价高,适合高精度场景);(5)检查 context 是否清晰带编号、最相关放最后。
10. (1)放宽边界——比如允许"基于资料推理"而不是"必须直接引用";(2)区分场景——主流问题正常答、敏感问题才拒答;(3)Few-shot 加一些"资料中虽然没明说但可以合理推断"的例子;(4)允许 LLM 在拒答时给出"可能相关的资料章节",而不是简单"我不知道"。

</details>

***

#### 5.11 本章总结 <a href="#id-511-ben-zhang-zong-jie" id="id-511-ben-zhang-zong-jie"></a>

<figure><img src=".gitbook/assets/总结 (1).jpg" alt=""><figcaption></figcaption></figure>
