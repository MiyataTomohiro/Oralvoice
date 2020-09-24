from flask import Flask 
import numpy as np
import librosa
import firebase_admin
import librosa.display
import matplotlib.pyplot as plt
from firebase_admin import storage
from firebase_admin import credentials
import matplotlib
# バックエンドを指定
matplotlib.use('Agg')

app = Flask(__name__)

def get_bucket():
    cred = credentials.Certificate(
        'test-f9b22-firebase-adminsdk-1okno-0c3413c7c5.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'test-f9b22.appspot.com'
    })
    bucket = storage.bucket()
    return bucket


def download_blob(bucket, source_blob_name, destination_file_name):
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))


def upload_blob(bucket, source_file_name, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))


def create_melspectrogram():
    "wavファイルのデータを短時間フーリエ変換してメルスペクトログラムを作成"
    data, fs = librosa.load("user.wav", sr=None)
    S = librosa.feature.melspectrogram(data, sr=fs, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(2.56, 2.56))
    librosa.display.specshow(log_S, sr=fs)
    # 枠と目盛りを消去
    plt.axis("off")
    # メルスペクトログラムを保存
    plt.savefig("user.png")
    plt.close()


@app.route("/")
def main():    
    bucket = get_bucket()
    download_blob(bucket, "test.wav", "user.wav")
    create_melspectrogram()
    upload_blob(bucket, "user.png", "test.png")
    x = np.array([1, 2, 3])
    print(x)
    y = np.append(x, [4, 5, 6])
    print(y)
    return "END"
