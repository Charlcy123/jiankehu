#!/usr/bin/env python3
"""
压缩录音，方便上传（audio-to-shorthand skill）

语音转写只需要 16kHz 单声道，把录音压到 24-32kbps 的 Opus/OGG 之后：
  1 小时 ≈ 11-14MB，3 小时 ≈ 35-40MB。超过 --max-mb 的会按时长自动切成几段。

用法：
  python3 shrink.py 录音.m4a                    # 输出 录音.shrunk.ogg（或多段 录音.part01.ogg ...）
  python3 shrink.py 录音.m4a --max-mb 28        # 每段不超过 28MB（默认 28，留出上传余量）
  python3 shrink.py 录音.m4a --bitrate 24k      # 更小，音质差一点，普通话仍可转写
  python3 shrink.py a.m4a b.mp3 --out ./小文件/

优先用系统 ffmpeg；没有 ffmpeg 就用 PyAV（pip install faster-whisper 时已经装上）。
压缩后的文件直接给 transcribe.py 或上传即可，时间戳按段自动带上偏移信息（见 .parts.json）。
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def bitrate_to_int(b: str) -> int:
    b = b.lower().strip()
    return int(float(b[:-1]) * 1000) if b.endswith("k") else int(b)


def probe_duration(path: Path) -> float:
    try:
        import av
        with av.open(str(path)) as c:
            if c.duration:
                return c.duration / av.time_base
    except Exception:
        pass
    if shutil.which("ffprobe"):
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)],
                             capture_output=True, text=True).stdout.strip()
        try:
            return float(out)
        except ValueError:
            pass
    return 0.0


def encode_ffmpeg(src: Path, dst: Path, start: float, length: float | None, bitrate: str):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src)]
    if length:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", bitrate,
            "-application", "voip", str(dst)]
    subprocess.run(cmd, check=True)


def encode_pyav(src: Path, dst: Path, start: float, length: float | None, bitrate: str):
    import av
    from av.audio.resampler import AudioResampler

    end = start + length if length else None
    with av.open(str(src)) as inp, av.open(str(dst), "w", format="ogg") as out:
        astream = next(s for s in inp.streams if s.type == "audio")
        ostream = out.add_stream("libopus", rate=16000)
        ostream.bit_rate = bitrate_to_int(bitrate)
        try:
            ostream.layout = "mono"
        except Exception:
            pass
        resampler = AudioResampler(format="s16", layout="mono", rate=16000)
        if start > 0:
            inp.seek(int(start / astream.time_base), stream=astream, backward=True, any_frame=False)
        for frame in inp.decode(astream):
            t = float(frame.pts * astream.time_base) if frame.pts is not None else 0.0
            if t < start:
                continue
            if end is not None and t >= end:
                break
            frame.pts = None   # 让编码器从 0 重新计时，切出来的每段时长才正确
            for rf in resampler.resample(frame):
                for pkt in ostream.encode(rf):
                    out.mux(pkt)
        for rf in resampler.resample(None):
            for pkt in ostream.encode(rf):
                out.mux(pkt)
        for pkt in ostream.encode(None):
            out.mux(pkt)


def shrink_one(src: Path, out_dir: Path, max_mb: float, bitrate: str, use_ffmpeg: bool):
    duration = probe_duration(src)
    if duration <= 0:
        print(f"✗ {src.name}：读不出时长，可能不是音频文件", file=sys.stderr)
        return False
    bps = bitrate_to_int(bitrate)
    est_mb = duration * bps / 8 / 1024 / 1024 * 1.3    # Opus 是变码率，留 30% 余量
    parts = max(1, math.ceil(est_mb / max_mb))
    part_len = duration / parts
    out_dir.mkdir(parents=True, exist_ok=True)
    encode = encode_ffmpeg if use_ffmpeg else encode_pyav

    print(f"→ {src.name}  时长 {duration/60:.1f} 分钟  原始 {src.stat().st_size/1024/1024:.1f}MB  "
          f"预计压缩后 {est_mb:.1f}MB → {'不切分' if parts == 1 else f'切 {parts} 段'}"
          f"（{'ffmpeg' if use_ffmpeg else 'PyAV'}）", file=sys.stderr)

    manifest = {"source": src.name, "duration_sec": round(duration, 2), "bitrate": bitrate, "parts": []}
    for i in range(parts):
        start = i * part_len
        length = part_len if parts > 1 else None
        dst = out_dir / (f"{src.stem}.part{i+1:02d}.ogg" if parts > 1 else f"{src.stem}.shrunk.ogg")
        br = bitrate
        for attempt in range(3):
            encode(src, dst, start, length, br)
            size_mb = dst.stat().st_size / 1024 / 1024
            if size_mb <= max_mb or attempt == 2:
                break
            br = f"{max(12, int(bitrate_to_int(br) * 0.7 / 1000))}k"
            print(f"   {dst.name} {size_mb:.1f}MB 超过 {max_mb}MB，降到 {br} 重编", file=sys.stderr)
        manifest["parts"].append({"file": dst.name, "offset_sec": round(start, 2),
                                  "length_sec": round(length or duration, 2), "size_mb": round(size_mb, 2),
                                  "bitrate": br})
        print(f"   ✓ {dst.name}  {size_mb:.1f}MB  起点 {int(start//60):02d}:{int(start%60):02d}", file=sys.stderr)
        if size_mb > max_mb:
            print(f"   ⚠ 降到 {br} 仍超过 {max_mb}MB，用 --max-mb 更小的值多切几段", file=sys.stderr)

    if parts > 1:
        mpath = out_dir / f"{src.stem}.parts.json"
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   分段清单: {mpath.name}（整理速记时把各段时间戳加上 offset_sec）", file=sys.stderr)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--max-mb", type=float, default=28)
    p.add_argument("--bitrate", default="32k", help="24k / 32k（默认）/ 48k")
    p.add_argument("--out", default=None, help="输出目录，默认与源文件同目录")
    p.add_argument("--no-ffmpeg", action="store_true", help="强制用 PyAV")
    args = p.parse_args()

    use_ffmpeg = bool(shutil.which("ffmpeg")) and not args.no_ffmpeg
    if not use_ffmpeg:
        try:
            import av  # noqa
        except ImportError:
            print("需要 ffmpeg 或 PyAV 之一：brew/apt install ffmpeg，或 pip install av", file=sys.stderr)
            sys.exit(1)

    ok = True
    for raw in args.inputs:
        src = Path(raw).expanduser()
        if not src.exists():
            print(f"✗ 找不到 {src}", file=sys.stderr)
            ok = False
            continue
        out_dir = Path(args.out) if args.out else src.parent
        try:
            ok &= shrink_one(src, out_dir, args.max_mb, args.bitrate, use_ffmpeg)
        except Exception as e:
            print(f"✗ {src.name} 压缩失败：{e}", file=sys.stderr)
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
