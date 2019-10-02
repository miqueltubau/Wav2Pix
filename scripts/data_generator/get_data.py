import csv
import youtube_dl as yt
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-url_csv", required=True,
                    help="CSV file where the url of the videos to download are.")
parser.add_argument("-dataset_path", required=True,
                    help="Output folder of the dataset divided into /video and /audio.")
args = parser.parse_args()


def read_channels(file_path):

    parsed_channels = []
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            parsed_channels.append({"name": row["Name"],
                                    "gender": row["Gender"],
                                    "url": row["Channel-URL"]})
    return parsed_channels

def download(url_list):
    error_counter = 0
    count = 0

    for url in url_list:
        count += 1
        print "Downloading videos and audios {}/{} with url [{}] from {}".format(count, len(url_list), url['url'], url['name'])
        out_path = os.path.join(args.dataset_path, url['name'])
        video_out_path = os.path.join(out_path, 'video')
        audio_out_path = os.path.join(out_path, 'audio')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            os.makedirs(video_out_path)
            os.makedirs(audio_out_path)

        try:
            video_options = {
                'format': "135, 140",
                'verbose': True,
                'continuedl': True,
                'ignoreerrors': True,
                'nooverwrites': True,
                'sleep_interval': 5,
                'playliststart': 1,
                'playlistend': 15,

            }

            with yt.YoutubeDL(video_options) as video_ydl:
                video_ydl.download([url['url']])


        except Exception:
            print "Download error."
            error_counter += 1


        # ffmpeg -i input_file -ss 00:00:15.00 -t 00:00:10.00 -c copy out.mp4
        workdir = os.listdir('./')
        for files in workdir:
            if files.endswith(".mp4"):
                shutil.move(files, video_out_path)
            elif files.endswith(".m4a"):
                shutil.move(files, audio_out_path)
    print "Found {} errors".format(error_counter)


urls = read_channels(args.url_csv)
download(urls)
