import argparse
import wave
import audioop
import subprocess as sp
import os
import unicodedata

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_path", required=True,
                    help="Parent directory where the dataset is stored")
args = parser.parse_args()
errors = 0


def format_filename(filename):
    try:
        filename = filename.decode('utf-8')
        s = ''.join((c for c in unicodedata.normalize('NFD', unicode(filename)) if unicodedata.category(c) != 'Mn'))
        return s.decode()
    except (UnicodeEncodeError, UnicodeDecodeError):
        return filename


def aac2wav(input_file, target_file, errors):
    try:
        command = 'ffmpeg -i ' + input_file + ' -vn ' + target_file
        print command
        sp.check_call(command, shell=True)
        return errors
    except (sp.CalledProcessError, IOError):
        errors += 1
        return errors


youtubers = os.listdir(args.dataset_path)
youtubers_dataset = [os.path.join(args.dataset_path, youtuber, 'audio')for youtuber in youtubers]

for youtuber_audio_path in youtubers_dataset:
    audio_files = os.listdir(youtuber_audio_path)
    for aac_audio_fname in audio_files:
        os.rename(os.path.join(youtuber_audio_path, aac_audio_fname),
                  os.path.join(youtuber_audio_path,
                               format_filename(aac_audio_fname.replace(" ", "").replace("'", "").replace('"', '').replace('(', '')
                                               .replace(')', '').replace('#', '').replace('&', '').replace(';', '').replace('!', '').
                                               replace(',','').replace('$', ''))))
        aac_audio_fname = format_filename(aac_audio_fname.replace(" ", "").replace("'", "").replace('"', '').replace('(', '')
                                               .replace(')', '').replace('#', '').replace('&', '').replace(';', '')
                                          .replace('!', '').replace(',','').replace('$', ''))
        print 'Processing {} file'.format(aac_audio_fname.encode('utf-8'))
        # Convert from AAC to WAV:
        if not aac_audio_fname.endswith(".wav"):
            wav_audio_fname = aac_audio_fname.replace(".m4a", ".wav")
        else:
            print 'Was already a wav file!!!'
            wav_audio_fname = aac_audio_fname #It is not AAC but WAV

        if not os.path.exists(os.path.join(youtuber_audio_path, wav_audio_fname)):
            errors = aac2wav(os.path.join(youtuber_audio_path, aac_audio_fname),
                    os.path.join(youtuber_audio_path, wav_audio_fname), errors)

        if not wav_audio_fname.endswith("_preprocessed.wav"):
            output_file = wav_audio_fname.replace(".wav", "_preprocessed.wav")
        else:
            output_file = wav_audio_fname

        if not os.path.exists(os.path.join(youtuber_audio_path, output_file)):
            #Read audio and obtain metadata:
            audio_reader = wave.open(os.path.join(youtuber_audio_path, wav_audio_fname), 'r')
            audio_writer = wave.open(os.path.join(youtuber_audio_path, output_file), 'w')
            nchannels, sampwidth, framerate, nframes, comptype, compname = audio_reader.getparams()
            data = audio_reader.readframes(nframes)
            try:
                #Downsample to 16kHz, 16 bits, mono:
                converted = audioop.ratecv(data, sampwidth, nchannels, framerate, 16000, None)
                if sampwidth is not 2:
                    print 'Converting sample width from {} to 2'.format(sampwidth)
                    converted = audioop.lin2lin(converted[0], sampwidth, 2)
                if nchannels is not 1:
                    converted = audioop.tomono(converted[0], 2, 1, 0)
            except Exception:
                print 'Failed to downsample wav file.'

            try:
                #Write output audio file:
                audio_writer.setparams((1, 2, 16000, 0, 'NONE', 'Uncompressed'))
                audio_writer.writeframes(converted)
            except Exception:
                print 'Failed to write wav file.'

            try:
                audio_reader.close()
                audio_writer.close()
            except Exception:
                print 'Failed to close wav files'

print 'Found {} errors'.format(errors)
