import os
import glob
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# 画像が保存されているルートディレクトリのパス
PATH = 'C:\\Users\\Miyata Tomohiro\\Desktop\\Oral Voice\\classifier\\ラベル\\'

# 分類するラベル
folder = ["positive", "negative"]
# 分類クラス数(今回は2種類)
num_classes = len(folder)
# 分類器を作成する単語
word_list = ["ask", "best", "day", "know", "live",
             "love", "right", "see", "think", "use"]
# 学習させる単語を選択
word = word_list[0]


def plot_history(history, save_graph_img_path,
                 fig_size_width, fig_size_height):
    """グラフを作成する関数"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # グラフ表示
    plt.figure(figsize=(fig_size_width, fig_size_height))  # ウィンドウ作成

    # 正解率グラフ
    plt.subplot(1, 2, 1)  # 2つ横に並べて右側に表示
    plt.plot(acc, label='acc', ls='-', marker='o')  # 学習用データのaccuracy
    plt.plot(val_acc, label='val_acc', ls='-', marker='x')  # 訓練用データのaccuracy
    plt.title('Training and validation accuracy')
    plt.xlabel('epoch')  # 横軸
    plt.ylabel('accuracy')  # 縦軸
    plt.legend(['acc', 'val_acc'])  # 凡例
    plt.grid(color='gray', alpha=0.2)  # グリッド表示

    # 損失グラフ
    plt.subplot(1, 2, 2)  # 2つ横に並べて左側に表示
    plt.plot(loss, label='loss', ls='-', marker='o')  # 学習用データのloss
    plt.plot(val_loss, label='val_loss', ls='-', marker='x')  # 訓練用データのloss
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'])
    plt.grid(color='gray', alpha=0.2)

    # 作成したグラフを図として保存
    plt.savefig(save_graph_img_path)
    plt.clf()
    plt.close()  # バッファ解放


def main():
    # ハイパーパラメータ
    batch_size = 32  # バッチサイズ
    epochs = 20  # エポック数(学習の繰り返し回数)
    dropout_rate = 0.2  # 過学習防止用：入力の20%を0にする（破棄）

    # データ格納用のディレクトリパス
    SAVE_DIR = "/Users/Miyata Tomohiro/Desktop/Oral Voice/classifier/"
    save_path = os.path.join(SAVE_DIR, word)
    os.makedirs(save_path, exist_ok=True)

    # グラフ画像のサイズ
    FIG_SIZE_WIDTH = 10
    FIG_SIZE_HEIGHT = 5

    # 画像データ用配列
    data_x = []
    # ラベルデータ用配列
    data_y = []

    # 画像を読み込む
    for label, img_title in enumerate(folder):
        if img_title == "positive":  # 正解ラベルの時
            file_dir = os.path.join(PATH, img_title, word)
        else:  # 不正解ラベルの時
            file_dir = os.path.join(PATH, img_title)
        img_file = glob.glob(file_dir + '/*.png')
        for i in img_file:
            img = img_to_array(load_img(i, target_size=(256, 256)))
            data_x.append(img)
            data_y.append(label)

    # Numpy配列を4次元リスト化
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    # 学習データはfloat32型に変換し、正規化(0～1)
    data_x = data_x.astype('float32') / 255.0
    # 正解ラベルをOne-hotにしたラベルに変換
    data_y = np_utils.to_categorical(data_y, num_classes)

    # 学習用データとテストデータに分割
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.2, stratify=data_y)

    # データセットの個数を表示
    print(x_train.shape, 'x train samples')
    print(x_test.shape, 'x test samples')
    print(y_train.shape, 'y train samples')
    print(y_test.shape, 'y test samples')

    # モデルの構築
    # CNN（畳み込みニューラルネットワーク）のモデルを設定
    model = Sequential()
    # 入力層:32×32*3
    # 【2次元畳み込み層】
    # 問題が複雑ならフィルタの種類を増やす
    # Conv2D：2次元畳み込み層で、画像から特徴を抽出（活性化関数：relu）
    # 入力データにカーネルをかける（「3×3」の32種類のフィルタを各マスにかける）
    # 出力ユニット数：32（32枚分の出力データが得られる）
    # input_shapeに関しては最初の層で、入力の形を指定しなければならない。
    # shape[1:]の書き方は、84x256x256だったら256x256が出力され、最初の値が無視される。
    # padding="same"は:の入力と同じ長さを出力がもつように入力にパディング
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:], activation='relu'))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

    # 【プーリング層】
    # 特徴量を圧縮する層。（ロバスト性向上、過学習防止、計算コスト抑制のため）
    # 畳み込み層で抽出された特徴の位置感度を若干低下させ、対象とする特徴量の画像内での位置が若干変化した場合でもプーリング層の出力が普遍になるようにする。
    # 画像の空間サイズの大きさを小さくし、調整するパラメーターの数を減らし、過学習を防止
    # pool_size=(2, 2):「2×2」の大きさの最大プーリング層。
    # 入力画像内の「2×2」の領域で最大の数値を出力。
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ドロップアウト(過学習防止用, dropout_rate=0.2なら20%のユニットを無効化）
    model.add(Dropout(dropout_rate))

    # 【2次元畳み込み層】
    # 問題が複雑ならフィルタの種類を増やす
    # 入力データにカーネルをかける（「3×3」の64種類のフィルタを使う）
    # 出力ユニット数：64（64枚分の出力データが得られる）
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    # 【2次元畳み込み層】
    # 問題が複雑ならフィルタの種類を増やす
    # 入力データにカーネルをかける（「3×3」の128種類のフィルタを使う）
    # 出力ユニット数：128（128枚分の出力データが得られる）
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    # 平坦化（次元削減）
    # 1次元ベクトルに変換
    model.add(Flatten())

    # 全結合層
    # 出力ユニット数：512
    model.add(Dense(512, activation='relu'))

    # ドロップアウト(過学習防止用, dropout_rate=0.2なら20%のユニットを無効化）
    model.add(Dropout(dropout_rate))

    # 全結合層
    # 2値分類なので、ユニット数2, 分類問題なので活性化関数はsoftmax関数
    # Softmax関数で総和が1となるように、各出力の予測確率を計算
    model.add(Dense(2, activation='sigmoid'))  # 活性化関数：softmax

    # コンパイル（2クラス分類問題）
    # 学習率:0.001、損失関数：binary_crossentropy、最適化アルゴリズム：Adam、評価関数：accuracy(正解率)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # 学習時間の計測開始
    start = time.time()

    # 早期停止
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='auto')
    # val_lossの改善が2エポック見られなかったら、学習率を0.5倍する。
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
    # 構築したモデルで学習（学習データ:trainのうち、10％を検証データ:validationとして使用）
    # verbose=1:標準出力にログを表示
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(x_test, y_test),
                        callbacks=[early_stopping, reduce_lr])

    # テスト用データセットで学習済分類器に入力し、パフォーマンスを計測
    score = model.evaluate(x_test, y_test, verbose=0)

    # 学習時間の計測終了(秒を分にする)
    elapsed_time = int((time.time() - start)/60)

    # パフォーマンス計測の結果を表示
    # 損失値（値が小さいほど良い）
    print(f'Test loss:{score[0]:.2f}')

    # 正答率（値が大きいほど良い）
    print(f'Test accuracy:{score[1]:.2f}')

    with open(save_path+'/CNN_学習記録.txt', mode='a') as f:
        text = f"バッチサイズ: {batch_size} エポック数: {epochs}"
        score = f"損失値: {score[0]:.2f} 正答率: {score[1]:.2f}"
        learning_time = f"学習時間: {elapsed_time}分"
        val = f"ドロップアウト: {dropout_rate}"
        f.write(f"{text} {score} {learning_time} {val}\n")

    # 学習過程をプロット
    plot_history(history, save_graph_img_path=save_path + "/graph.png",
                 fig_size_width=FIG_SIZE_WIDTH,
                 fig_size_height=FIG_SIZE_HEIGHT)

    # モデル構造の保存
    open(save_path + "/model.json", "w").write(model.to_json())

    # 学習済みの重みを保存
    model.save_weights(save_path + "/weight.hdf5")

    # 学習履歴を保存
    with open(save_path + "/history.json", 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    main()
