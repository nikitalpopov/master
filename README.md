# Master

Master's thesis for unsupervised online speaker diarization

## To start

```bash
$ sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg python-pyaudio python3-pyaudio
$ pip3 install git+https://github.com/Desklop/WebRTCVAD_Wrapper
$ pip3 install lxml SoundFile pyannote.core pyannote.audio pyannote.metrics scipy pydub numpy librosa sounddevice webrtcvad PyAudio
$ pip3 install --no-cache-dir Resemblyzer
$ mkdir results
$ mkdir chunks
```

## To prepare transcripts file for AMI recording

- select recording transcripts (`ami_corpus/ami_public_manual/words`, only ES2002a is left, everything is available at <http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip>)
- set correct variables in `data_preparation/__init__.py`
- run script to make transcripts.json

```bash
$ python3 data_preparation/__init__.py
```

## To run online diarization

```bash
$ python3 audio_recording/online.py
```

## To run file diarization

```bash
$ python3 audio_recording/file.py
```
