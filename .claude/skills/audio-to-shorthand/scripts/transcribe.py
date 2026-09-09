#!/usr/bin/env python3
"""
录音转写脚本（audio-to-shorthand skill）

用法：
  python3 transcribe.py --check                       # 检查环境
  python3 transcribe.py 录音.m4a                       # 自动选引擎和模型
  python3 transcribe.py 录音.m4a --hotwords "三联,张三" --out ./out
  python3 transcribe.py a.m4a b.m4a --engine faster-whisper --model large-v3
  python3 transcribe.py 录音.mp3 --engine openai       # 需要 OPENAI_API_KEY
  python3 transcribe.py 录音.mp3 --engine dashscope    # 需要 DASHSCOPE_API_KEY

每个输入文件输出三个文件到 --out 目录（默认音频同目录）：
  <名>.segments.json   结构化分段，含起止秒数和文本
  <名>.txt             每段一行，前缀 [mm:ss]，直接可作逐字稿
  <名>.srt             字幕格式，可导入剪辑软件核对

引擎：
  faster-whisper  本机运行，免费离线，首次使用会从 HuggingFace 下载模型。
                  依赖: pip install faster-whisper   （自带音频解码，不需要系统装 ffmpeg）
  openai          OpenAI 语音转写 API。依赖: pip install openai；环境变量 OPENAI_API_KEY。
                  单文件限 25MB，超过需要系统有 ffmpeg 用来切分。
  dashscope       阿里云百炼 Paraformer（对中文和方言友好）。依赖: pip install dashscope；
                  环境变量 DASHSCOPE_API_KEY。需要把音频上传到可公网访问的 URL，或用 --audio-url。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".mp4", ".amr", ".flac", ".ogg", ".opus",
              ".wma", ".mov", ".aac", ".webm", ".mkv", ".3gp", ".caf"}


# ---------- 通用工具 ----------

def fmt_ts(seconds: float, srt: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if srt:
        return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"
    if h:
        return f"{h:d}:{m:02d}:{int(s):02d}"
    return f"{m:02d}:{int(s):02d}"


def write_outputs(segments, src: Path, out_dir: Path, meta: dict):
    """segments: list of {"start": float, "end": float, "text": str}"""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    cleaned = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        cleaned.append({"start": round(float(seg["start"]), 2),
                        "end": round(float(seg["end"]), 2),
                        "text": text})

    json_path = out_dir / f"{stem}.segments.json"
    json_path.write_text(json.dumps({"meta": meta, "segments": cleaned},
                                    ensure_ascii=False, indent=2), encoding="utf-8")

    txt_lines = [f"# 逐字稿：{src.name}",
                 f"# 引擎：{meta.get('engine')}  模型：{meta.get('model')}  "
                 f"时长：{fmt_ts(meta.get('duration', 0))}  语言：{meta.get('language')}",
                 "# 机器原始转写，未经整理。时间戳为该段起始时间。", ""]
    txt_lines += [f"[{fmt_ts(s['start'])}] {s['text']}" for s in cleaned]
    txt_path = out_dir / f"{stem}.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    srt_blocks = []
    for i, s in enumerate(cleaned, 1):
        srt_blocks.append(f"{i}\n{fmt_ts(s['start'], True)} --> {fmt_ts(s['end'], True)}\n{s['text']}\n")
    srt_path = out_dir / f"{stem}.srt"
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    return json_path, txt_path, srt_path, cleaned


def probe_duration(path: Path) -> float:
    """用 PyAV 读时长（faster-whisper 的依赖，自带解码器）；没有就用 ffprobe；都没有返回 0。"""
    try:
        import av  # noqa
        with av.open(str(path)) as c:
            if c.duration:
                return c.duration / av.time_base
            for st in c.streams:
                if st.type == "audio" and st.duration and st.time_base:
                    return float(st.duration * st.time_base)
    except Exception:
        pass
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=nw=1:nk=1", str(path)],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
            return float(out)
        except Exception:
            pass
    return 0.0


def pick_model(duration: float, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    # 时长越长，错一处的代价越大；短音频用 small 快速出结果。
    if duration <= 0:
        return "medium"
    if duration < 15 * 60:
        return "small"
    if duration < 60 * 60:
        return "medium"
    return "large-v3"


def build_initial_prompt(hotwords: str | None) -> str:
    """Whisper 用 initial_prompt 引导风格和词表：给一句带标点的简体中文示例，再附上热词。"""
    base = "以下是普通话的对话记录，使用简体中文和中文标点。"
    if hotwords:
        words = [w.strip() for w in hotwords.replace("，", ",").split(",") if w.strip()]
        if words:
            base += "提到的名词包括：" + "、".join(words) + "。"
    return base


# ---------- 引擎：faster-whisper ----------

def run_faster_whisper(path: Path, model_name: str, language: str, hotwords: str | None,
                       device: str, verbose: bool):
    from faster_whisper import WhisperModel

    compute = "float16" if device == "cuda" else "int8"
    t0 = time.time()
    if verbose:
        print(f"  加载模型 {model_name}（首次使用会下载，几百 MB 到 3GB 不等）...", file=sys.stderr)
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute)
    except Exception as e:
        raise RuntimeError(
            f"模型 {model_name} 加载失败：{e}\n"
            "  常见原因：1) 无法访问 huggingface.co，可设置 HF_ENDPOINT=https://hf-mirror.com 后重试；"
            "2) 内存不够，换小一号模型 --model small；3) 指定了 --device cuda 但没有 GPU。") from e

    segments_iter, info = model.transcribe(
        str(path),
        language=None if language == "auto" else language,
        beam_size=5,
        vad_filter=True,                      # 跳过静音，减少静音段的"幻听"重复
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=build_initial_prompt(hotwords),
        condition_on_previous_text=False,     # 避免一段错了后面跟着错
    )
    segments = []
    last_print = 0
    for seg in segments_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
        if verbose and seg.end - last_print > 300:
            last_print = seg.end
            print(f"  已转写到 {fmt_ts(seg.end)} / {fmt_ts(info.duration)}", file=sys.stderr)
    meta = {"engine": "faster-whisper", "model": model_name, "language": info.language,
            "duration": info.duration, "elapsed_sec": round(time.time() - t0, 1)}
    return segments, meta


# ---------- 引擎：OpenAI ----------

def split_with_ffmpeg(path: Path, chunk_sec: int, tmpdir: Path):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("文件超过 25MB，需要系统安装 ffmpeg 才能切分后上传 OpenAI；"
                           "或者改用 --engine faster-whisper。")
    pattern = tmpdir / "chunk_%03d.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "segment",
                    "-segment_time", str(chunk_sec), "-ac", "1", "-ar", "16000",
                    "-b:a", "48k", str(pattern)], check=True)
    return sorted(tmpdir.glob("chunk_*.mp3"))


def run_openai(path: Path, model_name: str, language: str, hotwords: str | None, verbose: bool):
    from openai import OpenAI
    client = OpenAI()
    model_name = model_name if model_name not in (None, "auto") else "whisper-1"
    t0 = time.time()
    chunk_sec = 600
    files = [path]
    offset_map = {path: 0.0}
    tmpdir = None
    if path.stat().st_size > 24 * 1024 * 1024:
        tmpdir = Path(tempfile.mkdtemp(prefix="asr_"))
        files = split_with_ffmpeg(path, chunk_sec, tmpdir)
        offset_map = {f: i * chunk_sec for i, f in enumerate(files)}

    segments = []
    for f in files:
        if verbose:
            print(f"  上传 {f.name} ...", file=sys.stderr)
        with open(f, "rb") as fh:
            resp = client.audio.transcriptions.create(
                model=model_name, file=fh,
                language=None if language == "auto" else language,
                prompt=build_initial_prompt(hotwords),
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        off = offset_map[f]
        for seg in getattr(resp, "segments", None) or []:
            d = seg if isinstance(seg, dict) else seg.model_dump()
            segments.append({"start": d["start"] + off, "end": d["end"] + off, "text": d["text"]})
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    duration = probe_duration(path) or (segments[-1]["end"] if segments else 0)
    meta = {"engine": "openai", "model": model_name, "language": language,
            "duration": duration, "elapsed_sec": round(time.time() - t0, 1)}
    return segments, meta


# ---------- 引擎：DashScope (阿里云百炼 Paraformer) ----------

def run_dashscope(path: Path, model_name: str, language: str, hotwords: str | None,
                  audio_url: str | None, verbose: bool):
    import dashscope
    from dashscope.audio.asr import Transcription
    import urllib.request

    if not audio_url:
        raise RuntimeError("dashscope 引擎需要音频的公网 URL（Paraformer 录音文件识别是异步拉取的）。"
                           "把文件传到 OSS/对象存储后用 --audio-url 传入，或改用 faster-whisper。")
    model_name = model_name if model_name not in (None, "auto") else "paraformer-v2"
    t0 = time.time()
    task = Transcription.async_call(model=model_name, file_urls=[audio_url],
                                    language_hints=["zh", "en"] if language in ("zh", "auto") else [language])
    result = Transcription.wait(task=task.output.task_id)
    if result.status_code != 200:
        raise RuntimeError(f"dashscope 返回错误：{result.code} {result.message}")
    segments = []
    for item in result.output.get("results", []):
        url = item.get("transcription_url")
        if not url:
            continue
        data = json.loads(urllib.request.urlopen(url, timeout=60).read().decode("utf-8"))
        for tr in data.get("transcripts", []):
            for s in tr.get("sentences", []):
                segments.append({"start": s["begin_time"] / 1000, "end": s["end_time"] / 1000,
                                 "text": s["text"]})
    duration = probe_duration(path) or (segments[-1]["end"] if segments else 0)
    meta = {"engine": "dashscope", "model": model_name, "language": language,
            "duration": duration, "elapsed_sec": round(time.time() - t0, 1)}
    return segments, meta


# ---------- 环境检查 ----------

def check_env():
    print("== 录音转写环境检查 ==")
    ok_any = False
    try:
        import faster_whisper  # noqa
        import av  # noqa
        print("[✓] faster-whisper 已安装（本机离线转写，推荐）")
        ok_any = True
    except ImportError:
        print("[ ] faster-whisper 未安装 → pip install faster-whisper")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai  # noqa
            print("[✓] OpenAI：已配置 OPENAI_API_KEY，openai 包已安装")
            ok_any = True
        except ImportError:
            print("[ ] OpenAI：有 OPENAI_API_KEY 但缺 openai 包 → pip install openai")
    else:
        print("[ ] OpenAI：未设置 OPENAI_API_KEY（可选）")
    if os.environ.get("DASHSCOPE_API_KEY"):
        try:
            import dashscope  # noqa
            print("[✓] DashScope：已配置 DASHSCOPE_API_KEY，dashscope 包已安装")
            ok_any = True
        except ImportError:
            print("[ ] DashScope：有 DASHSCOPE_API_KEY 但缺 dashscope 包 → pip install dashscope")
    else:
        print("[ ] DashScope：未设置 DASHSCOPE_API_KEY（可选）")
    print(f"[{'✓' if shutil.which('ffmpeg') else ' '}] ffmpeg：{'已安装' if shutil.which('ffmpeg') else '未安装（faster-whisper 不需要；OpenAI 大文件切分需要）'}")
    try:
        import torch  # noqa
        gpu = torch.cuda.is_available()
        print(f"[{'✓' if gpu else ' '}] GPU：{'可用，加 --device cuda 会快很多' if gpu else '不可用，用 CPU（medium 模型约 1/3 实时速度）'}")
    except ImportError:
        print("[ ] GPU：未检测（没装 torch，不影响 CPU 转写）")
    if not ok_any:
        print("\n没有任何可用引擎。最简单的办法：pip install faster-whisper")
        return 1
    print("\n环境可用。首次转写会下载模型，请保持网络通畅；国内网络可先执行：export HF_ENDPOINT=https://hf-mirror.com")
    return 0


# ---------- 主流程 ----------

def transcribe_one(path: Path, args):
    duration = probe_duration(path)
    model_name = pick_model(duration, args.model) if args.engine in ("auto", "faster-whisper") else args.model
    print(f"→ {path.name}  时长 {fmt_ts(duration) if duration else '未知'}", file=sys.stderr)

    engines = [args.engine] if args.engine != "auto" else ["faster-whisper", "openai", "dashscope"]
    errors = []
    for eng in engines:
        try:
            if eng == "faster-whisper":
                segments, meta = run_faster_whisper(path, model_name, args.language, args.hotwords,
                                                    args.device, not args.quiet)
            elif eng == "openai":
                if args.engine == "auto" and not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError("未设置 OPENAI_API_KEY")
                segments, meta = run_openai(path, args.model, args.language, args.hotwords, not args.quiet)
            elif eng == "dashscope":
                if args.engine == "auto" and not os.environ.get("DASHSCOPE_API_KEY"):
                    raise RuntimeError("未设置 DASHSCOPE_API_KEY")
                segments, meta = run_dashscope(path, args.model, args.language, args.hotwords,
                                               args.audio_url, not args.quiet)
            else:
                raise RuntimeError(f"未知引擎 {eng}")
            break
        except ImportError as e:
            errors.append(f"{eng}: 缺少依赖 {e.name}")
        except Exception as e:
            errors.append(f"{eng}: {e}")
    else:
        print(f"✗ {path.name} 转写失败：", file=sys.stderr)
        for err in errors:
            print(f"   - {err}", file=sys.stderr)
        return None

    if args.engine == "auto" and errors and not args.quiet:
        print("  （已跳过的引擎：" + "；".join(errors) + "）", file=sys.stderr)

    meta["source"] = path.name
    meta["hotwords"] = args.hotwords or ""
    if not meta.get("duration"):
        meta["duration"] = duration
    out_dir = Path(args.out) if args.out else path.parent
    json_path, txt_path, srt_path, cleaned = write_outputs(segments, path, out_dir, meta)
    print(f"✓ {path.name}  {len(cleaned)} 段  引擎 {meta['engine']}/{meta['model']}  "
          f"用时 {meta.get('elapsed_sec', '?')}s", file=sys.stderr)
    print(f"   逐字稿: {txt_path}\n   分段:   {json_path}\n   字幕:   {srt_path}", file=sys.stderr)
    if not cleaned:
        print("   ⚠ 没有识别出任何内容，检查音频是否有声音、格式是否正常。", file=sys.stderr)
    return txt_path


def main():
    p = argparse.ArgumentParser(description="录音转写（audio-to-shorthand skill）",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("inputs", nargs="*", help="音频/视频文件，可多个")
    p.add_argument("--check", action="store_true", help="只检查环境，不转写")
    p.add_argument("--engine", default="auto",
                   choices=["auto", "faster-whisper", "openai", "dashscope"])
    p.add_argument("--model", default="auto",
                   help="faster-whisper: tiny/base/small/medium/large-v3（默认按时长自动）；"
                        "openai: whisper-1 等；dashscope: paraformer-v2 等")
    p.add_argument("--language", default="zh", help="zh / en / yue(粤语) / auto，默认 zh")
    p.add_argument("--hotwords", default=None, help="专有名词，逗号分隔，提高识别准确率")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="faster-whisper 运行设备")
    p.add_argument("--audio-url", default=None, help="dashscope 引擎用：音频的公网 URL")
    p.add_argument("--out", default=None, help="输出目录，默认音频同目录")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.check:
        sys.exit(check_env())
    if not args.inputs:
        p.error("请给出至少一个音频文件，或用 --check 检查环境")

    paths = []
    for raw in args.inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"✗ 找不到文件：{path}", file=sys.stderr)
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            print(f"⚠ {path.name} 后缀不像音频文件，仍然尝试转写", file=sys.stderr)
        paths.append(path)
    if not paths:
        sys.exit(1)

    failed = 0
    for path in paths:
        if transcribe_one(path, args) is None:
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
