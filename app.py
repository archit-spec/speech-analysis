
import spaces
import gradio as gr
import pandas as pd
import yt_dlp
import os
from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import HuggingFaceEncoder
from faster_whisper import WhisperModel
import io

# Function to download YouTube audio and return it as a BytesIO object
def download_youtube_audio(url, preferred_quality="192"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': preferred_quality,
        }],
        'outtmpl': '-',  # Output to stdout
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', None)
            print(f"Downloading audio for: {video_title}")

            # Download audio to a BytesIO object
            audio_buffer = io.BytesIO()
            ydl.download([url], audio_buffer)
            audio_buffer.seek(0)
            print("Audio download complete")
            return audio_buffer

    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading audio: {e}")
        return None

# Function to transcribe audio from BytesIO using WhisperModel
@spaces.GPU
def transcribe(audio_buffer, model_name="medium"):
    model = WhisperModel(model_name)
    print("Reading audio buffer")
    
    # Hypothetical support for BytesIO object
    segments, info = model.transcribe(audio_buffer)
    return segments

# Function to process segments and convert them into a DataFrame
@spaces.GPU
def process_segments(segments):
    result = {}
    print("Processing...")
    for i, segment in enumerate(segments):
        chunk_id = f"chunk_{i}"
        result[chunk_id] = {
            'chunk_id': segment.id,
            'chunk_length': segment.end - segment.start,
            'text': segment.text,
            'start_time': segment.start,
            'end_time': segment.end
        }
    df = pd.DataFrame.from_dict(result, orient='index')
    df.to_csv('final.csv')  # Save DataFrame to final.csv
    return df

# Gradio interface functions
@spaces.GPU
def generate_transcript(youtube_url, model_name="large-v3"):
    audio_buffer = download_youtube_audio(youtube_url)
    if audio_buffer is None:
        return "Error downloading audio"
    
    segments = transcribe(audio_buffer, model_name)
    df = process_segments(segments)
    
    lis = list(df['text'])
    encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")
    chunker = StatisticalChunker(encoder=encoder, dynamic_threshold=True, min_split_tokens=30, max_split_tokens=40, window_size=2, enable_statistics=False)
    chunks = chunker._chunk(lis)
    
    row_index = 0
    for i in range(len(chunks)):
        for j in range(len(chunks[i].splits)):
            df.at[row_index, 'chunk_id2'] = f'chunk_{i}'
            row_index += 1
    
    grouped = df.groupby('chunk_id2').agg({
        'start_time': 'min',
        'end_time': 'max',
        'text': lambda x: ' '.join(x),
        'chunk_id': list
    }).reset_index()
    
    grouped = grouped.rename(columns={'chunk_id': 'chunk_ids'})
    grouped['chunk_length'] = grouped['end_time'] - grouped['start_time']
    grouped['chunk_id'] = grouped['chunk_id2']
    grouped = grouped.drop(columns=['chunk_id2', 'chunk_ids'])
    grouped.to_csv('final.csv')
    df = pd.read_csv("final.csv")
    transcripts = df.to_dict(orient='records')
    
    return transcripts

# Function to download video using yt-dlp and generate transcript HTML
def download_video(youtube_url):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': 'downloaded_video.mp4',
        'quiet': True
    }

    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_path = 'downloaded_video.mp4'

    if not os.path.exists(video_path):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

    transcripts = generate_transcript(youtube_url)
    transcript_html = ""
    for t in transcripts:
        transcript_html += f'<div class="transcript-block"><a href="#" onclick="var video = document.getElementById(\'video-player\').querySelector(\'video\'); video.currentTime={t["start_time"]}; return false;">' \
                           f'[{t["start_time"]:.2f} - {t["end_time"]:.2f}]<br>{t["text"]}</a></div>'
    
    return video_path, transcript_html

# Function to search the transcript
def search_transcript(keyword):
    transcripts = pd.read_csv("final.csv").to_dict(orient='records')
    search_results = ""
    for t in transcripts:
        if keyword.lower() in t['text'].lower():
            search_results += f'<div class="transcript-block"><a href="#" onclick="var video = document.getElementById(\'video-player\').querySelector(\'video\'); video.currentTime={t["start_time"]}; return false;">' \
                              f'[{t["start_time"]:.2f} - {t["end_time"]:.2f}]<br>{t["text"]}</a></div>'
    return search_results

# CSS for styling
css = """
.fixed-video { width: 480px !important; height: 270px !important; }
.fixed-transcript { width: 480px !important; height: 270px !important; overflow-y: auto; }
.transcript-block { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }
.transcript-block a { text-decoration: none; color: #007bff; }
.transcript-block a:hover { text-decoration: underline; }
"""

# Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# YouTube Video Player with Clickable Transcript")

    with gr.Row():
        youtube_url = gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video link here")
        download_button = gr.Button("Download and Display Transcript")
    
    with gr.Row():
        video = gr.Video(label="Video Player", elem_id="video-player", elem_classes="fixed-video")
        transcript_display = gr.HTML(label="Transcript", elem_classes="fixed-transcript")

    with gr.Row():
        search_box = gr.Textbox(label="Search Transcript", placeholder="Enter keyword to search")
        search_button = gr.Button("Search")
        search_results_display = gr.HTML(label="Search Results", elem_classes="fixed-transcript")

    # On button click, download the video and display the transcript
    def display_transcript(youtube_url):
        video_path, transcript_html = download_video(youtube_url)
        return video_path, transcript_html

    download_button.click(fn=display_transcript, inputs=youtube_url, outputs=[video, transcript_display])

    # On search button click, search the transcript and display results
    search_button.click(fn=search_transcript, inputs=search_box, outputs=search_results_display)

# Launch the interface
demo.launch()
