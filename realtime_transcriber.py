#!/usr/bin/env python3
import argparse
import queue
import threading
import sys
import time
from collections import deque
from datetime import datetime
import numpy as np
import sounddevice as sd
import whisper
import torch

class RealtimeTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        chunk_duration: float = 3.0,
        sample_rate: int = 16000,
        refinement_history: int = 3,
    ):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.refinement_history = max(1, refinement_history)
        self.chunk_history: deque[np.ndarray] = deque(maxlen=self.refinement_history)
        self.last_refined_text: str = ""
        self._live_line_active = False
        self._live_line_length = 0
        
        # Auto-select device if not provided (prioritize MPS for M1/M2 Macs)
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        print(f"Loading Whisper model '{model_size}' on device '{device}'...")
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        
        # Audio buffer
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]
        
        self.audio_queue.put(audio_data.copy())
    
    def get_system_audio_device(self):
        """Find system audio device for macOS"""
        devices = sd.query_devices()
        
        # Look for system audio devices (typically contain "BlackHole" or "Soundflower" on macOS)
        system_devices = []
        for i, device in enumerate(devices):
            name = device['name'].lower()
            if ('blackhole' in name or 'soundflower' in name or 
                'loopback' in name or 'virtual' in name):
                if device['max_input_channels'] > 0:
                    system_devices.append((i, device))
        
        if system_devices:
            device_id, device_info = system_devices[0]
            print(f"Using system audio device: {device_info['name']}")
            return device_id
        else:
            print("No virtual audio device found. Install BlackHole or Soundflower for system audio capture.")
            print("Using default input device instead...")
            return None
    
    def start_recording(self, use_system_audio: bool = True):
        """Start recording audio"""
        device_id = None
        if use_system_audio:
            device_id = self.get_system_audio_device()
        
        self.is_recording = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            device=device_id,
            samplerate=self.sample_rate,
            channels=1 if device_id is None else None,
            callback=self.audio_callback,
            blocksize=1024,
        )
        
        self.stream.start()
        print(f"Recording started... (sample rate: {self.sample_rate} Hz)")
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self._clear_live_line()
        print("Recording stopped.")
    
    def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a chunk of audio"""
        try:
            # Ensure audio is float32 and normalized
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_chunk)) > 0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
            
            # Transcribe with Whisper
            fp16 = self.device in ["cuda", "mps"]
            result = self.model.transcribe(
                audio_chunk,
                fp16=fp16,
                language=None,  # Auto-detect
                task="transcribe"
            )
            
            text = result.get("text", "").strip()
            return text
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def _diff_refined_text(self, refined_text: str) -> tuple[str, bool]:
        """Return the portion of text that is new compared to the last refinement."""
        if not refined_text:
            return "", False
        
        if not self.last_refined_text:
            self.last_refined_text = refined_text
            return refined_text, False
        
        last_text = self.last_refined_text
        common_prefix_length = 0
        for c1, c2 in zip(last_text, refined_text):
            if c1 != c2:
                break
            common_prefix_length += 1
        
        # Avoid cutting through a word when possible
        if (
            common_prefix_length
            and common_prefix_length < len(refined_text)
            and not refined_text[common_prefix_length - 1].isspace()
        ):
            last_space = refined_text.rfind(" ", 0, common_prefix_length)
            if last_space != -1:
                common_prefix_length = last_space + 1
            else:
                common_prefix_length = 0
        
        new_text = refined_text[common_prefix_length:].strip()
        is_refinement = common_prefix_length < len(last_text)
        self.last_refined_text = refined_text
        return new_text, is_refinement
    
    def _print_live_line(self, timestamp: str, text: str) -> None:
        """Show immediate transcription while waiting for refinement."""
        line = f"[{timestamp}] (live) {text}"
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        self._live_line_active = True
        self._live_line_length = len(line)
    
    def _replace_live_line(self, timestamp: str, text: str, is_refinement: bool) -> None:
        """Replace the live line with the refined result."""
        if not text.strip():
            self._clear_live_line()
            return
        
        prefix = "Refined: " if is_refinement else ""
        line = f"[{timestamp}] {prefix}{text}"
        if self._live_line_active:
            sys.stdout.write("\r" + " " * self._live_line_length + "\r")
            sys.stdout.flush()
        print(line, flush=True)
        self._live_line_active = False
        self._live_line_length = 0
    
    def _clear_live_line(self) -> None:
        """Clear any live line when no refined text is produced."""
        if not self._live_line_active:
            return
        sys.stdout.write("\r" + " " * self._live_line_length + "\r")
        sys.stdout.flush()
        self._live_line_active = False
        self._live_line_length = 0
    
    def process_audio(self):
        """Process audio chunks from the queue"""
        audio_buffer = np.array([], dtype=np.float32)
        self.chunk_history.clear()
        self.last_refined_text = ""
        self._live_line_active = False
        self._live_line_length = 0
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio data from queue (with timeout)
                audio_data = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, audio_data])
                
                # Process when we have enough audio
                if len(audio_buffer) >= self.chunk_size:
                    # Extract chunk
                    chunk = audio_buffer[:self.chunk_size]
                    audio_buffer = audio_buffer[self.chunk_size:]
                    
                    # Check if chunk has enough audio content
                    if np.max(np.abs(chunk)) > 0.01:  # Minimum volume threshold
                        chunk = chunk.astype(np.float32, copy=True)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        live_text = self.transcribe_chunk(chunk)
                        if live_text:
                            self._print_live_line(timestamp, live_text)
                        else:
                            self._clear_live_line()
                        
                        self.chunk_history.append(chunk)
                        combined_audio = np.concatenate(list(self.chunk_history))
                        text = self.transcribe_chunk(combined_audio)
                        
                        if text:
                            new_text, is_refinement = self._diff_refined_text(text)
                            if self._live_line_active:
                                final_text = new_text if new_text else (live_text or "")
                                self._replace_live_line(timestamp, final_text, is_refinement)
                            elif new_text:
                                prefix = "Refined: " if is_refinement else ""
                                print(f"[{timestamp}] {prefix}{new_text}")
                        else:
                            if self._live_line_active:
                                self._replace_live_line(timestamp, live_text or "", False)
                    
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                break
    
    def run(self, use_system_audio: bool = True):
        """Run the real-time transcriber"""
        try:
            self.start_recording(use_system_audio)
            
            # Start processing thread
            process_thread = threading.Thread(target=self.process_audio)
            process_thread.daemon = True
            process_thread.start()
            
            print("Real-time transcription started. Press Ctrl+C to stop...")
            print("=" * 60)
            
            # Keep main thread alive
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping transcription...")
        finally:
            self.stop_recording()
            # Wait a bit for the processing thread to finish
            time.sleep(1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription using OpenAI Whisper"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Force device selection (default: auto)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Audio chunk duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--refinement-chunks",
        type=int,
        default=3,
        help="Number of recent audio chunks to use for refinement (default: 3)",
    )
    parser.add_argument(
        "--use-mic",
        action="store_true",
        help="Use microphone instead of system audio",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    transcriber = RealtimeTranscriber(
        model_size=args.model,
        device=args.device,
        chunk_duration=args.chunk_duration,
        sample_rate=args.sample_rate,
        refinement_history=args.refinement_chunks,
    )
    
    use_system_audio = not args.use_mic
    transcriber.run(use_system_audio=use_system_audio)

if __name__ == "__main__":
    main()
