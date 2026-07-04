import argparse
import os
import sys
import time
import re
import tempfile
import subprocess
import numpy as np

import soundfile as sf
from voxcpm import VoxCPM

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "openbmb", "VoxCPM2")

# 每段最大字符数，超长文本自动分段避免累积漂移
MAX_CHARS_PER_SEG = int(os.environ.get("VOXCPM_MAX_CHARS", "50"))


def split_sentences(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r'(?<=[。！？.!?\n])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]
    segments = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) > max_chars and buf:
            segments.append(buf.strip())
            buf = s
        else:
            buf += s
    if buf:
        segments.append(buf.strip())
    return segments


def main():
    parser = argparse.ArgumentParser(
        description="VoxCPM2 声音克隆"
    )
    parser.add_argument("--text", "-t", required=True, help="要合成的文本")
    parser.add_argument("--output", "-o", default="output.wav", help="输出音频路径")
    parser.add_argument("--device", default="auto", help="运行设备 (auto/mps/cuda/cpu)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ref-audio", "-ra", help="参考音频路径 (可控克隆，免文本)")
    group.add_argument("--prompt-audio", "-pa", help="提示音频路径 (终极克隆)")

    parser.add_argument("--prompt-text", "-pt", help="提示音频的对应文本")
    parser.add_argument("--cfg", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--steps", type=int, default=10, help="推理步数")
    parser.add_argument("--no-denoiser", action="store_true", help="禁用降噪器")
    parser.add_argument("--no-optimize", action="store_true", help="禁用 torch.compile 优化")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--no-play", action="store_true", help="生成后不播放")

    args = parser.parse_args()

    if args.prompt_audio and not args.prompt_text:
        parser.error("--prompt-audio 需要同时提供 --prompt-text")
    if args.prompt_text and not args.prompt_audio:
        parser.error("--prompt-text 需要同时提供 --prompt-audio")

    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)

    print(f"加载模型: {MODEL_PATH}")
    print(f"设备: {args.device}", flush=True)

    t0 = time.time()
    model = VoxCPM(
        voxcpm_model_path=MODEL_PATH,
        enable_denoiser=True,
        optimize=not args.no_optimize,
        device=args.device,
    )
    print(f"模型加载完成 ({time.time() - t0:.1f}s)", flush=True)

    sample_rate = model.tts_model.sample_rate
    print(f"采样率: {sample_rate} Hz")

    if args.ref_audio:
        print(f"可控克隆模式 (参考音频: {args.ref_audio})")
    elif args.prompt_audio:
        print(f"终极克隆模式 (提示音频: {args.prompt_audio})")
    else:
        print("普通 TTS 模式")

    # 参考音频预处理：用 ffmpeg anlmdn 降噪
    preproc_ref = None
    if args.ref_audio:
        preproc_ref = tempfile.mktemp(suffix="_preprocessed.wav")
        print(f"参考音频降噪...", flush=True)
        subprocess.run([
            "ffmpeg", "-y", "-i", args.ref_audio,
            "-af", "anlmdn",
            "-ar", "16000", "-ac", "1",
            preproc_ref
        ], check=True, capture_output=True)

    # 分段：长文本自动拆分，避免累积漂移
    texts = split_sentences(args.text, MAX_CHARS_PER_SEG)
    if len(texts) > 1:
        print(f"文本较长，自动分为 {len(texts)} 段生成（每段 ≤{MAX_CHARS_PER_SEG} 字）")

    all_wavs = []
    for i, seg_text in enumerate(texts):
        if len(texts) > 1:
            print(f"  第 {i+1}/{len(texts)} 段...", flush=True)

        t1 = time.time()
        wav = model.generate(
            text=seg_text,
            reference_wav_path=preproc_ref or args.ref_audio,
            prompt_wav_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            cfg_value=args.cfg,
            inference_timesteps=args.steps,
            denoise=not args.no_denoiser,
        )
        elapsed = time.time() - t1
        dur = len(wav) / sample_rate
        all_wavs.append(wav)

        if len(texts) > 1:
            print(f"     -> {dur:.1f}s ({elapsed:.1f}s)", flush=True)

    if preproc_ref and os.path.exists(preproc_ref):
        os.unlink(preproc_ref)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if len(all_wavs) > 1:
        combined = np.concatenate(all_wavs)
        sf.write(args.output, combined, sample_rate)
        total_dur = len(combined) / sample_rate
        total_gen = time.time() - t0
        print(f"输出: {args.output}")
        print(f"总时长: {total_dur:.1f}s | 生成: {total_gen:.1f}s | RTF: {total_gen/total_dur:.2f}")
    else:
        wav = all_wavs[0]
        sf.write(args.output, wav, sample_rate)
        duration = len(wav) / sample_rate
        gen_time = time.time() - t0
        print(f"输出: {args.output}")
        print(f"时长: {duration:.2f}s | 生成: {gen_time:.1f}s | RTF: {gen_time/duration:.2f}")

    if not args.no_play:
        print(f"正在播放: {args.output}", flush=True)
        subprocess.run(["afplay", args.output])


if __name__ == "__main__":
    main()
