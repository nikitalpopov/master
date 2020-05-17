# Master
## To start

```bash
$ pip3 install -r requirements.txt
$ mkdir results
$ mkdir chunks
```

## To prepare transcripts file for AMI recording

- select recording transcripts (ami_corpus -> ami_public_manual -> words, only ES are left, everything is available at <http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip)>
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
