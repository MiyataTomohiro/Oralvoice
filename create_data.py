import os  # ディレクトリを移動するために使う
import glob  # パス付でファイル名を取得するために使う
import librosa
import subprocess  # ffmpeg.exeを実行するためのモジュール
import numpy as np
import librosa.display
import soundfile as sf
import speech_recognition as sr
import matplotlib.pyplot as plt

# 使用する動画があるフォルダ名
folder = ["ask", "best", "close", "day", "diary", "english", "eye", "floor",
          "glad", "guitar", "hat", "important", "just", "know", "live",
          "love", "mine", "never", "often", "pick", "plane", "right", "room",
          "see", "think", "until", "use", "will", "yet", "zoom", "テストデータ"]

# 分類器を作成する単語(正解画像になる)
positive = ["ask", "best", "day", "know", "live",
            "love", "right", "see", "think", "use"]


# 入力先ディレクトリの設定
DIR_PATH = "C:\\Users\\Miyata Tomohiro\\Desktop\\Oral Voice\\data\\file\\"
# 出力先ディレクトリの設定
PATH = "C:\\Users\\Miyata Tomohiro\\Desktop\\Oral Voice\\classifier\\ラベル\\"


def mp4_wav(word, *file):
    "動画(mp4)ファイルを音声(wav)ファイルに変換する"
    # splitを使ってファイル名と拡張子に分けてファイル名だけをnameに入れる
    name = [n.split(".")[0] for n in file]
    basename = os.path.splitext(os.path.basename(name[0]))[0]
    # 使用するパスを設定
    output = os.path.join(DIR_PATH, "wav", word)
    os.makedirs(output, exist_ok=True)
    input_path = os.path.join(DIR_PATH, "mp4", word, basename+".mp4")
    output_path = os.path.join(output, basename+".wav")
    # \"はエスケープシーケンス
    """
    cmd = f'ffmpeg.exe -y -i "{input_path}" -ac 1 "{output_path}"'
    subprocess.call(cmd)  # コマンドを実行
    # 音声データを読み込んで無音区間(1dB以下)を探す
    wav, sr = librosa.load(output_path, sr=None)
    # data, _ = librosa.effects.trim(wav, top_db=40, ref=np.mean)
    time = librosa.effects.split(wav, top_db=30)
    # 無音が終わった瞬間をスタート位置にする(サンプル数を時間にする)
    start = librosa.samples_to_time(time[0, 1], sr=sr)
    # スタート位置から音声データを読み込む(開始位置を揃える)
    data, sr = librosa.load(output_path, sr=None, offset=start)
    # 音声ファイルにする(音声データを書き込む)
    sf.write(output_path, data, sr)
    """
    create_waveform(word, output_path, basename)


def create_waveform(name, path, fname):
    "wavファイルのデータを数値化してグラフ化"
    data, _ = librosa.load(path, sr=None)
    # グラフの描画先の準備
    fig = plt.figure(figsize=(10.24, 2.56))

    # 音声データをグラフ化
    plt.plot(data)

    # グラフを画像として保存
    fig.set_size_inches
    save_path = os.path.join(DIR_PATH, "waveform", name)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{fname}.png"))
    plt.clf()
    plt.close(fig)
    # create_spectrogram(data, path, name, fname)
    # create_spectrogram(path, name, fname)
    # create_mfcc(path, name, fname)
    create_melspectrogram(path, name, fname)


"""
def create_spectrogram(data, path, label, fname):
    "スペクトログラムを作成する"
    wf = wave.open(path, "r")  # グラフ化するwavファイルを開く

    # FFTのサンプル数
    N = 512

    # FFTで用いるハミング窓
    hammingWindow = np.hamming(N)

    # スペクトログラムを描画
    with np.errstate(divide='ignore'):  # withスコープ内のみ制御を適用する
        plt.specgram(data, NFFT=N, Fs=wf.getframerate(),
                     noverlap=0, window=hammingWindow)
    # 枠と目盛りを消去
    plt.axis("off")

    # スペクトログラムを保存
    save_path = os.path.join(DIR_PATH, "spectrogram", label)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))
    # 分類器で使用する画像の保存先を指定
    if label in positive:  # 正解画像の処理
        save_path = os.path.join(PATH, "positive", label)
    else:  # 不正解画像の処理(テストデータではない時)
        save_path = os.path.join(PATH, "negative")

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))
    plt.clf()
    plt.close()
"""


def create_spectrogram(path, label, fname):
    "スペクトログラムを作成する"
    y, sr = librosa.load(path, sr=None)
    D = librosa.stft(y)  # 短時間フーリエ変換する
    S, _ = librosa.magphase(D)  # 複素数を強度と位相へ変換
    Sdb = librosa.amplitude_to_db(S)  # 強度をdb単位へ変換
    # スペクトログラムを表示
    librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='log')
    # 枠と目盛りを消去
    plt.axis("off")

    # スペクトログラムを保存

    save_path = os.path.join(DIR_PATH, "spectrogram2", label)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))

    # 分類器で使用する画像の保存先を指定
    if label in positive:  # 正解画像の処理
        save_path = os.path.join(PATH, "positive", label)
    else:  # 不正解画像の処理(テストデータではない時)
        save_path = os.path.join(PATH, "negative")

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))
    plt.clf()
    plt.close()


def create_mfcc(path, label, fname):
    "メル周波数ケプストラムを作成する"
    y, sr = librosa.load(path, sr=None)
    # MFCCを算出
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, dct_type=3)
    librosa.display.specshow(mfcc, sr=sr)
    # 枠と目盛りを消去
    plt.axis("off")

    # メル周波数ケプストラムを保存
    save_path = os.path.join(DIR_PATH, "mfcc", label)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))

    # 分類器で使用する画像の保存先を指定
    if label in positive:  # 正解画像の処理
        save_path = os.path.join(PATH, "positive", label)
    else:  # 不正解画像の処理(テストデータではない時)
        save_path = os.path.join(PATH, "negative")

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))
    plt.clf()
    plt.close()


def create_melspectrogram(file_path, label, fname):
    "wavファイルのデータを短時間フーリエ変換してメルスペクトログラムを作成"
    data, fs = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(data, sr=fs, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(5.12, 5.12))
    librosa.display.specshow(log_S, sr=fs, x_axis='time', y_axis='mel')
    # 枠と目盛りを消去
    plt.axis("off")

    # メルスペクトログラムを保存
    save_path = os.path.join(DIR_PATH, "melspectrogram", label)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))

    # 分類器で使用する画像の保存先を指定
    if label in positive:  # 正解画像の処理
        save_path = os.path.join(PATH, "positive", label)
    elif label == "テストデータ":  # テストデータの時
        """
        r = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            voice = r.record(source)
        try:
            print(r.recognize_google(voice))
        except sr.UnknownValueError:
            print("could not understand audio")
        except sr.RequestError as e:
            print("Could not request results")
        """
        save_path = os.path.join(PATH, label)
    else:  # 不正解画像の処理(テストデータではない時)
        save_path = os.path.join(PATH, "negative")

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{fname}.png"))
    plt.clf()
    plt.close()


if __name__ == "__main__":
    for i in range(len(folder)):
        # 使用する動画のパス
        file_path = os.path.join(DIR_PATH, "mp4", folder[i], "*.mp4")

        # 正規表現でmp4ファイルだけを取得
        dir = glob.glob(file_path)

        # mp4ファイルをwavファイルにする
        for j in range(len(dir)):
            mp4_wav(folder[i], dir[j])
