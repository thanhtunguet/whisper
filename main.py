import argparse
from datetime import timedelta
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    td = timedelta(milliseconds=millis)
    # Ensure hours always shown in SRT
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    ms = millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments: list[dict], srt_path: Path) -> None:
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"]) if seg.get("start") is not None else "00:00:00,000"
        end = format_timestamp(seg["end"]) if seg.get("end") is not None else start
        text = (seg.get("text") or "").strip()
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def transcribe(
    audio_path: Path,
    model_size: str = "small",
    language: str | None = None,
    device: str | None = None,
) -> tuple[str, list[dict]]:
    import whisper  # lazy import to avoid dependency if only extracting

    # Auto-select device if not provided
    if device is None:
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    model = whisper.load_model(model_size, device=device)
    fp16 = device == "cuda"
    result = model.transcribe(str(audio_path), language=language, fp16=fp16)

    text: str = result.get("text", "").strip()
    segments: list[dict] = result.get("segments", [])
    return text, segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate subtitles using OpenAI Whisper from an audio file."
    )
    parser.add_argument("audio", type=str, help="Path to input audio file")
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ],
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., en, vi). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force device selection (default: auto)",
    )
    parser.add_argument(
        "--out-base",
        type=str,
        default=None,
        help="Output base path without extension. Defaults to audio stem next to input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if args.out_base is not None:
        out_base = Path(args.out_base).expanduser().resolve()
    else:
        out_base = audio_path.with_suffix("")

    text, segments = transcribe(
        audio_path=audio_path,
        model_size=args.model,
        language=args.language,
        device=args.device,
    )

    txt_path = out_base.with_suffix(".txt")
    srt_path = out_base.with_suffix(".srt")

    txt_path.write_text(text + "\n", encoding="utf-8")
    write_srt(segments, srt_path)

    print(str(srt_path))


if __name__ == "__main__":
    main()


