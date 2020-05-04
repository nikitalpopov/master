from operator import itemgetter
from lxml import etree
import xml.etree.ElementTree as ET
import json

def parse_ami_transcript_xml(file, speaker_id):
    """
    Parsing AMI transcript XML file
    """

    xmlp = ET.XMLParser(encoding="ISO-8859-1")
    f = ET.parse(file, parser=xmlp)

    # with open(file) as f:
    #     xml = f.read()

    root = f.getroot()

    transcript = []

    for element in list(root):
        # print(element)
        if element.tag == 'w':
            if element.text:
                text = element.text
            else:
                text = ''

            if element.get('punc'):
                punc = True
            else:
                punc = False

            transcript.append({
                'start': float(element.get('starttime')),
                'end': float(element.get('endtime')),
                'text': text,
                'punc': punc,
                'speaker_id': speaker_id
            })

    previous = transcript[0]
    for index, elem in enumerate(transcript):
        if elem['start'] == previous['end']:
            previous['end'] = elem['end']
            previous['text'] += ('' if elem['punc'] else ' ') + elem['text']
            transcript[index] = None
        else:
            del elem['punc']
            previous = elem

    transcript = [t for t in transcript if t and len(t['text'])]

    return transcript


if __name__ == "__main__":
    speakers_phrases = []

    for index, speaker in enumerate(['A', 'B', 'C', 'D']):
        t = parse_ami_transcript_xml(
            "ami-corpus/ES2002a.{}.words.xml".format(speaker), index)
        speakers_phrases.extend(t)

    final = sorted(speakers_phrases, key=itemgetter('start'))

    with open('ami-corpus/transcripts.json', 'w') as f:
        json.dump(final, f)
