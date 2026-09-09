# 录音转速记（audio-to-shorthand）

把采访、客户会议、内部会议、讲座的录音整理成速记稿。

## 怎么用

1. 把这个仓库（或 `.claude/skills/audio-to-shorthand` 文件夹）放进 Claude Code 的项目里，或者把打包好的 `.skill` 文件上传到 claude.ai 的技能里
2. 第一次用，让 Claude 检查环境：`python3 .claude/skills/audio-to-shorthand/scripts/transcribe.py --check`。缺什么它会告诉你，通常只需要 `pip install faster-whisper`
3. 之后把录音文件丢给 Claude，说"转速记"就行。已经有讯飞/飞书妙记转写文本的，直接丢文本

## 会得到什么

- `<文件名>-逐字稿.txt`：机器原始转写，带时间戳，用来核对
- `<文件名>-速记.md`：要点摘要 / 决定与待办 / 按话题分节的速记正文 / 可引用原话 / 存疑清单

## 文件

```
audio-to-shorthand/
├── SKILL.md              ← Claude 读的主说明
├── scripts/transcribe.py ← 转写脚本（faster-whisper 本机 / OpenAI / 阿里 DashScope）
├── references/整理规范.md ← 逐字稿怎么整理成速记，含前后对照例子
├── references/engines.md ← 引擎安装、模型选择、排错
└── assets/速记模板.md    ← 速记稿的固定结构
```
