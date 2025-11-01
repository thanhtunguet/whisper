import argparse
import shutil
import subprocess
from pathlib import Path


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg and ensure it is on your PATH."
        )


def build_output_path(input_path: Path, output: str | None, ext: str) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    return input_path.with_suffix(f".{ext}")


def extract_audio(
    input_video: Path,
    output_audio: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
    overwrite: bool = False,
) -> None:
    ensure_ffmpeg_available()

    if output_audio.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_audio}. Use --overwrite to replace."
        )

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_video),
        "-vn",
        "-acodec",
        codec,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_audio),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to extract audio:\n" + completed.stderr.strip()
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract audio from a video file using ffmpeg."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input video file (e.g., .mp4, .mov)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output audio path. Defaults to input stem with .wav",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="wav",
        choices=["wav", "flac", "mp3", "m4a"],
        help="Output audio extension/format (default: wav)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output audio sample rate (default: 16000)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of audio channels (default: 1)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="pcm_s16le",
        help="Audio codec to use (default: pcm_s16le for WAV)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_video = Path(args.input).expanduser().resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_audio = build_output_path(input_video, args.output, args.ext)

    extract_audio(
        input_video=input_video,
        output_audio=output_audio,
        sample_rate=args.sample_rate,
        channels=args.channels,
        codec=args.codec,
        overwrite=args.overwrite,
    )

    print(str(output_audio))


if __name__ == "__main__":
    main()


