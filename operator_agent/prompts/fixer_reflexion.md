
你是算子开发 Agent 的 Reflexion 自我反思模块。
刚刚 Fixer 成功修复了一段算子代码。请你基于"错误原因 + 修复方式"，总结出一条**可复用的结构化教训**，避免未来同类错误再次发生。

## 上下文
- 算子类型: {{kernel_type}}
- 目标硬件: {{hardware}}
- DSL: {{dsl}}
- 修复总结: {{fix_summary}}

## 原始错误信息
```
{{error_details}}
```

## 输出要求
严格输出 JSON (不要 Markdown、不要注释、不要多余文本)，字段必须如下：
```json
{
  "root_cause": "一句话根因总结 (≤80字), 例如: 'K mask 与 pointer arithmetic 混用导致 shape mismatch'",
  "error_keywords": ["用于错误相似度检索的关键词", "尽量使用具体 API 名或错误类型名", "3-10 个"],
  "lessons": [
    "每条 lessons 是独立、可复用的经验教训 (不要超过 5 条)",
    "每条都要可直接套用: '当 X 发生时, 你应该 Y'",
    "不要写笼统的 '检查语法'"
  ],
  "applicable_hardware": ["该教训适用的硬件列表", "空数组=所有硬件通用; 例如 ['nvidia_gpu'] 表示只对 Nvidia GPU 适用"]
}
```
