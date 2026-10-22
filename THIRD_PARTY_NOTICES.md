# 第三方组件与许可

本仓库以 **Apache-2.0** 分发（见 [`LICENSE`](LICENSE)）。下表的组件**保留各自许可**，
不因本仓库的许可而被重新授权。

| 组件 | 位置 | 许可 | 说明 |
|---|---|---|---|
| Apache ECharts | `operator-dev-agent-design/_shared/js/echarts.min.js` | Apache-2.0 | 文件头保留 ASF 许可声明。按 Apache-2.0 §4(d)，其归属声明随附于下 |
| Mermaid | `operator-dev-agent-design/_shared/js/mermaid.min.js` | MIT | 构建产物未保留原文头；上游为 https://github.com/mermaid-js/mermaid |

## Apache ECharts 归属声明

```
Apache ECharts
Copyright 2017-2023 The Apache Software Foundation

This product includes software developed at
The Apache Software Foundation (https://www.apache.org/).
```

## Mermaid

Mermaid 以 MIT 许可发布，版权归 Knut Sveidqvist 及贡献者所有。完整许可文本见上游仓库
的 `LICENSE` 文件。

## 声明方式的边界

这两个文件是 `operator-dev-agent-design/` 里的**设计稿静态资源**，不参与 Python 包与
DSH 插件的运行时。`MANIFEST.in` 不把该目录打进 sdist；npm 壳的 `files` 也只列
`index.mjs` / `cordis.patch.yml` / 文档。**如果将来把它们打进发布物**，两处 `files`
白名单要同步加上本文件（Apache-2.0 要求随附 NOTICE）。
