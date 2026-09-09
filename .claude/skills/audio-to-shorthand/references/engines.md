# 转写引擎说明

`scripts/transcribe.py` 支持三个引擎，`--engine auto`（默认）按下面顺序尝试，哪个能用用哪个。

## 1. faster-whisper（默认，推荐）

本机运行，免费、离线、不上传音频。对普通话效果好，对中英混杂尚可，对方言弱。

**安装**
```bash
pip install faster-whisper
```
它自带音频解码（PyAV），**不需要系统安装 ffmpeg**。m4a、mp3、mp4 都能直接读。

**模型选择**（`--model`，不指定则按时长自动）

| 模型 | 大小 | CPU 速度（相对实时） | 适合 |
|------|------|-------------------|------|
| tiny / base | 75MB / 145MB | 很快 | 只用来测试环境，中文效果差 |
| small | 480MB | 约 1x | 15 分钟以内的短录音，快速出结果 |
| medium | 1.5GB | 约 1/3 x | 一小时以内，准确率和速度的平衡点 |
| large-v3 | 3GB | 约 1/6 x（CPU）| 长录音、多人、有专有名词，或用户要求准确优先 |

有 NVIDIA GPU 加 `--device cuda`，large-v3 也能接近实时。Mac 上没有 CUDA，用 CPU 跑 medium 就好。

**首次使用会下载模型**，从 huggingface.co。国内网络下载不动时：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
然后重跑。模型缓存在 `~/.cache/huggingface/`，下过一次不用再下。

**热词**：`--hotwords "三联,生活周刊,张三"`。Whisper 没有真正的热词功能，脚本是把它们塞进 initial_prompt 里引导模型，对人名和品牌名有帮助，但不是百分百。转写完还是要人工对一遍专有名词。

**已知问题**
- 静音段会"幻听"出重复句或训练数据残留（"字幕由xx提供"、"请订阅"）。脚本已开 VAD 过滤减轻这个问题，剩下的整理时删掉并在存疑里注明
- 粤语用 `--language yue`，效果一般；其他方言基本不行，建议换 dashscope
- 两人以上同时说话的段落会丢内容

## 2. openai（OpenAI 语音转写 API）

要联网，按时长收费，音频会上传到 OpenAI。中文效果和 large-v3 相当，速度快很多。

```bash
pip install openai
export OPENAI_API_KEY=sk-...
python3 transcribe.py 录音.m4a --engine openai
```
单文件限 25MB。超过的话脚本会用系统 ffmpeg 切成 10 分钟一段再上传，所以大文件需要装 ffmpeg（`brew install ffmpeg` / `apt install ffmpeg`）。切段后时间戳会自动加偏移量。

`--model` 默认 `whisper-1`，也可以指定其他支持 `verbose_json` 分段输出的模型。

## 3. dashscope（阿里云百炼 Paraformer）

对中文、方言、中英混杂最友好，也是国内网络最稳的选项。要联网，按时长收费。

```bash
pip install dashscope
export DASHSCOPE_API_KEY=sk-...
python3 transcribe.py 录音.m4a --engine dashscope --audio-url "https://your-bucket.oss-cn-xxx.aliyuncs.com/录音.m4a"
```
Paraformer 的录音文件识别是**异步拉取**模式，需要音频有公网可访问的 URL（放 OSS 或任何对象存储都行）。没有 URL 的话用不了这个引擎，脚本会提示。本地文件参数仍要给，用来读时长和命名输出。

## 引擎都用不了的时候

按这个顺序建议用户：

1. 手机自带的录音 App 大多有转文字（iOS 备忘录、小米/华为录音机）
2. 讯飞听见、飞书妙记、腾讯会议的转写，导出 txt 或 srt
3. 剪映的"识别字幕"，导出 srt

拿到文本后直接进整理步骤。srt 文件有时间戳，txt 通常没有——没有时间戳的速记稿在基本信息里注明。

## 输出文件说明

| 文件 | 用途 |
|------|------|
| `<名>.txt` | 每段一行，前缀 `[mm:ss]`。这就是逐字稿，复制为 `<原文件名>-逐字稿.txt` |
| `<名>.segments.json` | 带精确起止秒的分段和元信息（引擎、模型、时长、热词）。需要程序化处理时用 |
| `<名>.srt` | 字幕格式。导入剪辑软件或播放器可以边听边看，核对存疑处最方便 |

## 排错

| 现象 | 处理 |
|------|------|
| `模型加载失败：403/ConnectionError` | 访问不了 huggingface.co，设置 `HF_ENDPOINT=https://hf-mirror.com` |
| `Killed` 或内存不足 | 模型太大，换 `--model small` 或 `medium` |
| 转出来全是英文或乱码 | 语言识别错了，明确加 `--language zh` |
| 大段重复同一句话 | 静音段幻听，整理时删除；音频开头有长静音可以先剪掉 |
| 没识别出任何内容 | 检查文件是否真的有声音、是否是 DRM 保护的格式 |
| 转写很慢 | 正常，CPU 跑 medium 一小时录音要 20-40 分钟。放后台跑，或换 small，或用云端引擎 |
