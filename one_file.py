import yt_dlp
import os


def download_audio(youtube_url, output_dir="audios", codec=None):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(title)s [%(id)s] - {codec if codec else "best"}.%(ext)s',
        'noplaylist': True,
    }

    if codec:
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': codec,
            'preferredquality': 'best' if codec in ['wav', 'flac'] else '192',  # best for lossless, 192k for lossy
        }]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


# Download in different formats
formats = [None, 'wav', 'flac', 'm4a', 'mp3']  # None for best original
for fmt in formats:
    download_audio(r"https://www.youtube.com/watch?v=VFzr53VJunU", codec=fmt)
