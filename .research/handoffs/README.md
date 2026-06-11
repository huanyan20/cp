# Handoff 協議（Cursor ↔ 外部工具）

跨軟體 agent 的**唯一通訊匯流排**。禁止依賴 chat 轉述。

## 檔案

| 路徑 | 方向 | 格式 |
|------|------|------|
| `handoffs/P8.json` | external → 全員 | implement 完成 |
| `handoffs/P10.json` | Cursor → 全員 | implement 完成 |
| `reviews/P8-reviewed-by-cursor.md` | Cursor → external | cross-review |
| `reviews/P10-reviewed-by-external.md` | external → Cursor | cross-review |

## Phase 機

```text
implement → handoff → cross_review → integrate
         ↑              ↑
    各寫各 branch    互換審查（唯讀 + 小修）
```

## Handoff JSON 必填欄位

| 欄位 | 說明 |
|------|------|
| `task_id` | `P8` / `P10` |
| `phase` | 固定 `"handoff"` |
| `agent` | `"cursor"` / `"external"` |
| `tool` | handoff 時填 `"antigravity-ide"`（P8）或 `"cursor"`（P10） |
| `branch` | git 分支名 |
| `pytest` | `"pass"` 才允許進 cross_review |
| `ready_for_cross_review` | `true` |

## Cross-review 規則

- 最多改 **3 個檔**；否則只寫 review memo，由 implementer 改
- 不得改對方 task 的核心假設
- merge 僅 **人類** 執行

## 範本

見 `.research/EXTERNAL_AGENT_BRIEF.md` §4、§5。
