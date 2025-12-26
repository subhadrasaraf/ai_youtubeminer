# ai_youtubeminer
Python script that takes a YouTube URL (e.g., a specific podcast episode), and does three things automatically: Downloads the audio. Uses VAD (Voice Activity Detection) to chop it into clean 30-second chunks (removing silence/music). Transcribes one chunk using a distinct Open Source model(Whisper)  and compares it to the YouTube auto-captions.
