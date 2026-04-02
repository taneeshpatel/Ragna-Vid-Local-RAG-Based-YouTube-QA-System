from downloader.audio_downloader import download_audio

url = input("Paste a YouTube video URL: ")
audio_file = download_audio(url)

if audio_file:
    print(f"✅ Audio saved to: {audio_file}")
else:
    print("❌ Audio download failed.")
