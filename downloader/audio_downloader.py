# import os
# from yt_dlp import YoutubeDL

# def download_audio(url: str, output_path: str = "cache/audio/audio.mp3") -> str:
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'outtmpl': output_path,
#         'ffmpeg_location': 'C:/ffmpeg-7.1.1-essentials_build/bin',  # 🛠️ Set here
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'mp3',
#             'preferredquality': '192',
#         }],
#         'quiet': True,
#         'noplaylist': True
#     }

#     try:
#         with YoutubeDL(ydl_opts) as ydl:
#             ydl.download([url])
#         return output_path
#     except Exception as e:
#         print(f"[Error] Failed to download audio: {e}")
#         return None


import os
import sys
import glob
import shutil
import hashlib
from yt_dlp import YoutubeDL

# Ensure utf-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def find_ffmpeg() -> str | None:
    """Finds the directory containing ffmpeg.exe if not in PATH."""
    if shutil.which("ffmpeg"):
        return None

    # Search WinGet package install directory
    winget_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_dir):
        matches = glob.glob(os.path.join(winget_dir, "**", "ffmpeg.exe"), recursive=True)
        if matches:
            return os.path.dirname(matches[0])

    # Check common hardcoded locations
    for candidate in [
        r"C:\ffmpeg\bin",
        r"C:\ffmpeg-7.1.1-essentials_build\bin",
    ]:
        if os.path.exists(os.path.join(candidate, "ffmpeg.exe")):
            return candidate

    return None


def download_audio(url: str, cache_dir: str = "cache/audio") -> str:
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a unique filename based on URL hash
    video_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    output_base = os.path.join(cache_dir, video_hash)
    output_path = f"{output_base}.mp3"

    # Remove existing stale audio
    if os.path.exists(output_path):
        try:
            print(f"[Downloader] Removing old cached audio: {output_path}")
            os.remove(output_path)
        except Exception:
            pass

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{output_base}.%(ext)s",  # will be converted to .mp3
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'noplaylist': True
    }

    ffmpeg_dir = find_ffmpeg()
    if ffmpeg_dir:
        ydl_opts['ffmpeg_location'] = ffmpeg_dir

    try:
        print(f"[Downloader] Downloading audio for: {url}")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Validate saved file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10_000:
            print(f"[Downloader] Audio downloaded successfully: {output_path}")
            return output_path
        else:
            print(f"[Downloader] Download failed or file too small: {output_path}")
            return None

    except Exception as e:
        print(f"[Downloader] Exception during download: {e}")
        return None
