import pyaudio
import soundfile as sf
from pydub import AudioSegment
import pydub.scipy_effects
from webrtcvad_wrapper import VAD
import pydub
import scipy.signal as sps
from scipy.spatial.distance import pdist, squareform, euclidean, cosine
from scipy.sparse import csgraph
import scipy
from pyannote.metrics.diarization import DiarizationErrorRate, GreedyDiarizationErrorRate, JaccardErrorRate
from pyannote.core import Segment, Timeline, Annotation, notebook
import webrtcvad
import sounddevice as sd
from numpy import linalg as LA
import numpy as np
import librosa
from resemblyzer import preprocess_wav, VoiceEncoder
import os
from timeit import default_timer as timer
import collections
import json
import datetime
from pathlib import Path
from math import ceil
import sys
from itertools import groupby
import itertools
flatten = itertools.chain.from_iterable

AudioSegment.converter = '/usr/local/Cellar/ffmpeg/4.2.2_2/bin/ffmpeg'


SENSITIVITY_MODE = 3  # 4 is supposed to be better for neural networks, but it's not

## Define Voice Activity Detector
vad = VAD(sensitivity_mode=SENSITIVITY_MODE)


def get_vad_segments(audio_segment):
    filtered_segments = vad.filter(audio_segment)

    return [
        (filtered_segment[0], filtered_segment[1])
        for filtered_segment in filtered_segments if filtered_segment[-1]
    ]


def remove_overlap(ranges):
    result = []
    current_start = -1
    current_stop = -1

    for start, stop in sorted(ranges):
        if start > current_stop:
            # this segment starts after the last segment stops
            # just add a new segment
            result.append((start, stop))
            current_start, current_stop = start, stop
        else:
            # segments overlap, replace
            result[-1] = (current_start, stop)
            # current_start already guaranteed to be lower
            current_stop = max(current_stop, stop)

    return result


def get_speaker_segments(labels_list):
    N = np.array(labels_list)
    counter = np.arange(1, np.alen(N))
    groupings = np.split(N, counter[N[1:] != N[:-1]])

    segments = []
    start = 0
    for group in groupings:
        if len(group) > 0 and group[0] is not None:
            segments.append({
                'start': start,
                'end': start + len(group) - 1,
                'speaker_id': group[0]
            })
        start += len(group)

    return segments


def get_similarity(embed_i, embed_j):
    return cosine(embed_i, embed_j)


def get_hypothesis(speaker_segments):
    hypothesis = Annotation()
    for t in speaker_segments:
        try:
            hypothesis[Segment(x[t['start']], x[t['end']])] = str(
                t['speaker_id']) + '_hyp'
        except:
            pass

    return hypothesis


der = DiarizationErrorRate()
gder = GreedyDiarizationErrorRate()
jer = JaccardErrorRate()


def measure_metrics(reference, hypothesis):
    der_value = der(reference, hypothesis)
    print('DER:', der_value)
    print('JER:', jer(reference, hypothesis))

    return der_value


def prepare_segment(filename, high_filter_value, low_filter_value):
    audio_segment = AudioSegment.from_file(filename)

    if audio_segment.sample_width != 2:
        audio_segment = audio_segment.set_sample_width(2)
    if audio_segment.channels != 1:
        audio_segment = audio_segment.set_channels(1)

    ## Apply 3rd order pass filters with given value
    audio_segment = audio_segment.high_pass_filter(
        high_filter_value, order=3)
    audio_segment = audio_segment.low_pass_filter(
        low_filter_value, order=3)

    return audio_segment


def run_file_diarization():
    SOURCE_FOLDER = 'records'
    SOURCE_FILE = 'Pumpkin_and_Honey_Bunny.wav'
    TRANSCRIPTS = f'{SOURCE_FOLDER}/transcripts.json'
    N_SPEAKERS = 3

    FILEPATH = str(Path(SOURCE_FOLDER, SOURCE_FILE))
    SR = 32000

    CHUNKS_FOLDER = 'chunks'
    RESULTS_FOLDER = 'results'

    CUT_AUDIO = True
    CUT_LENGTH = 10. * 60.  # seconds

    ## Load audio
    if CUT_AUDIO:
        wav, source_sr = librosa.load(
            FILEPATH, sr=SR, offset=0.0, duration=CUT_LENGTH)
    else:
        wav, source_sr = librosa.load(FILEPATH, sr=SR)

    ## Load transcripts from prepared file and mark reference annotations

    with open(TRANSCRIPTS, 'r') as f:
        ami_corpus_transcripts = json.load(f)

    if CUT_AUDIO and CUT_LENGTH < len(wav):
        ami_corpus_transcripts = [
            t for t in ami_corpus_transcripts if t['start'] < CUT_LENGTH]

    reference = Annotation()
    for t in ami_corpus_transcripts:
        try:
            reference[Segment(t['start'], t['end'])] = str(
                t['speaker_id']) + '_ref'
        except:
            pass

    timers = []

    ## Load voice encoder
    encoder = VoiceEncoder()

    CHUNK_TIME_LENGTH = 2.5  # seconds
    CHUNK_SIZE = ceil(CHUNK_TIME_LENGTH * SR)
    CHUNKS_OVERLAP = .15  # share (percents)
    SPEAKER_CHANGE_THRESHOLD = .3  # percents
    HIGH_FILTER_VALUE = 100
    LOW_FILTER_VALUE = 7000

    ## Clean folders
    for folder in [CHUNKS_FOLDER, RESULTS_FOLDER]:
        filelist = [f for f in os.listdir(folder)]
        for f in filelist:
            os.remove(os.path.join(folder, f))

    embeds = []
    speakers_ids = [0]
    speaker_segments = []
    vad_segments = []

    counter = 0

    ## Process overlapping chunks
    for i in range(0, len(wav), ceil(CHUNK_SIZE * (1 - CHUNKS_OVERLAP))):
        t_start = timer()

        start = i
        end = i + CHUNK_SIZE if i + CHUNK_SIZE < len(wav) - 1 else len(wav) - 1

        chunk = wav[start:end]
        chunk = preprocess_wav(chunk)

        if (len(chunk) > 0):
            filename = f'chunks/chunk_{i}.wav'
            sf.write(filename, chunk, samplerate=SR)
            audio_segment = prepare_segment(
                filename, HIGH_FILTER_VALUE, LOW_FILTER_VALUE)


            segments = get_vad_segments(audio_segment)

            vad_segments += [(start + ceil(segment[0] * SR), start +
                            ceil(segment[1] * SR)) for segment in segments]

            segments = [s for s in remove_overlap(vad_segments) if s[1] - s[0] > 0]
            segments = remove_overlap(vad_segments)
            segments = [(s[0] - start, s[1] - start)
                        for s in segments if s[0] >= start]

            for segment in segments:
                seg_start, seg_end = segment
                frame = chunk[seg_start:seg_end]
                embed = encoder.embed_utterance(frame)

                if len(embeds) > 0:
                    distances = []
                    for speaker_id, speaker_embeddings in groupby(sorted(embeds, key=lambda x: x[1]), lambda x: x[1]):
                        es = [e for e, s_id in list(speaker_embeddings)]

                        mean_dist = np.mean(
                            [get_similarity(embed, embedding) for embedding in es])
                        min_dist = np.min(
                            [get_similarity(embed, embedding) for embedding in es])
                        max_dist = np.max(
                            [get_similarity(embed, embedding) for embedding in es])

                        distances.append((min_dist, speaker_id))

                    distances = sorted(distances, key=lambda x: x[0])

                    if len(distances) > 0:
                        if distances[0][0] < SPEAKER_CHANGE_THRESHOLD:
                            id = distances[0][1]
                        else:
                            counter += 1
                            id = counter
                    else:
                        counter += 1
                        id = counter

                    speakers_ids.append(id)

                    speaker_segments.append({
                        'start': start + seg_start,
                        'end': start + seg_end,
                        'speaker_id': id
                    })
                else:
                    id = counter
                    speaker_segments.append({
                        'start': start + seg_start,
                        'end': start + seg_end,
                        'speaker_id': id
                    })

                embeds.append((embed, id))

        t_end = timer()
        timers.append(t_end - t_start)
    print(np.mean(timers))

    for speaker_id, segs in groupby(sorted(speaker_segments, key=lambda x: x['speaker_id']), lambda x: x['speaker_id']):
        print(f"Speaker {speaker_id}")
        audio = np.concatenate([wav[s['start']:s['end']] for s in segs], axis=0)
        print(audio.shape)
        filename = f"{RESULTS_FOLDER}/speaker_{speaker_id}.wav"
        sf.write(filename, audio, samplerate=SR)


def run_online_diarization():
    audio = []

    SR = 32000
    CHUNKS_FOLDER = 'chunks'
    RESULTS_FOLDER = 'results'

    ## Load voice encoder
    encoder = VoiceEncoder()

    CHUNK_TIME_LENGTH = 2.5  # seconds
    CHUNK_SIZE = ceil(CHUNK_TIME_LENGTH * SR)
    CHUNKS_OVERLAP = .15  # share (percents)
    SPEAKER_CHANGE_THRESHOLD = .3  # percents
    HIGH_FILTER_VALUE = 200
    LOW_FILTER_VALUE = 4000

    ## Clean folders
    for folder in [CHUNKS_FOLDER, RESULTS_FOLDER]:
        filelist = [f for f in os.listdir(folder)]
        for f in filelist:
            os.remove(os.path.join(folder, f))

    FORMAT = pyaudio.paFloat32
    CHANNELS = 1

    pa = pyaudio.PyAudio()

    # start Recording
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SR,
        input=True,
        frames_per_buffer=100
    )

    audio = []
    chunk_container = []
    embeds = []
    speakers_ids = [0]
    speaker_segments = []
    vad_segments = []

    counter = 0
    start = 0
    id = 0
    i = 0

    try:
        while True:
            data = stream.read(100, exception_on_overflow=False)
            recorded = np.array(np.fromstring(data, dtype=np.float32))

            audio = np.append(audio, recorded)
            chunk_container = np.append(chunk_container, recorded)

            if (len(chunk_container) >= CHUNK_SIZE):
                chunk = chunk_container.copy()
                chunk_container = []
                chunk = preprocess_wav(chunk)
                counter += 1

                if (len(chunk) > 0):
                    filename = f'{CHUNKS_FOLDER}/chunk_{i}.wav'
                    sf.write(filename, chunk, samplerate=SR)
                    audio_segment = prepare_segment(
                        filename, HIGH_FILTER_VALUE, LOW_FILTER_VALUE)

                    segments = get_vad_segments(audio_segment)

                    vad_segments += [(start + ceil(segment[0] * SR), start +
                                    ceil(segment[1] * SR)) for segment in segments]

                    segments = [s for s in remove_overlap(
                        vad_segments) if s[1] - s[0] > .2 * SR]
                    segments = [(s[0] - start, s[1] - start)
                                for s in segments if s[0] >= start]

                    for segment in segments:
                        seg_start, seg_end = segment
                        frame = chunk[seg_start:seg_end]
                        embed = encoder.embed_utterance(frame)

                        if len(embeds) > 0:
                            distances = []
                            for speaker_id, speaker_embeddings in groupby(sorted(embeds, key=lambda x: x[1]), lambda x: x[1]):
                                es = [e for e, s_id in list(speaker_embeddings)]

                                mean_dist = np.mean(
                                    [get_similarity(embed, embedding) for embedding in es])
                                min_dist = np.min(
                                    [get_similarity(embed, embedding) for embedding in es])
                                max_dist = np.max(
                                    [get_similarity(embed, embedding) for embedding in es])

                                distances.append((min_dist, speaker_id))

                            distances = sorted(distances, key=lambda x: x[0])

                            if len(distances) > 0:
                                if distances[0][0] < SPEAKER_CHANGE_THRESHOLD:
                                    id = distances[0][1]
                                else:
                                    counter += 1
                                    id = counter
                            else:
                                counter += 1
                                id = counter

                            speakers_ids.append(id)

                            speaker_segments.append({
                                'start': start + seg_start,
                                'end': start + seg_end,
                                'speaker_id': id
                            })
                        else:
                            id = counter
                            speaker_segments.append({
                                'start': start + seg_start,
                                'end': start + seg_end,
                                'speaker_id': id
                            })

                        embeds.append((embed, id))
                        print(f'{datetime.datetime.now()}: Speaker {id} was speaking now')
                i += 1
                chunk_container = audio[-ceil(CHUNK_SIZE *
                                              CHUNKS_OVERLAP):].copy()
                start = len(audio)-ceil(CHUNK_SIZE * CHUNKS_OVERLAP)

    except KeyboardInterrupt:
        print('Recording has finished')

        for speaker_id, segs in groupby(sorted(speaker_segments, key=lambda x: x['speaker_id']), lambda x: x['speaker_id']):
            audio_speaker = np.concatenate([audio[s['start']:s['end']]
                                            for s in segs], axis=0)
            filename = f'{RESULTS_FOLDER}/speaker_{speaker_id}.wav'
            sf.write(filename, audio_speaker, samplerate=SR)

        sf.write(f'{RESULTS_FOLDER}/audio_record.wav', audio, samplerate=SR)
        stream.stop_stream()
        stream.close()
        pa.terminate()

        pass

    except Exception as e:
        exit(type(e).__name__ + ': ' + str(e))



if __name__ == '__main__':
    # run_file_diarization()
    run_online_diarization()
