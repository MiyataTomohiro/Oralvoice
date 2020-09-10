# -*- coding: utf-8 -*-
from common import *

import tensorflow as tf

application = Flask(__name__)
application.config.from_object(__name__)

##################################################################
# 初期メニュー
##################################################################
@application.route('/')
def index():
    return render_template('index.html')

#各画面からの戻りの時はこちら(/でうまく戻れないので) ※indexとIndexをわざと変えている        
@application.route('/Index',methods=['POST'])
def Index():
    return render_template('index.html')


#**************************************
# アップロード
#**************************************
#アップロード画面初期表示
@application.route('/tensorUpload',methods=['POST'])
def tensorUpload():   
    return render_template('Upload.html')



#***********************************************************************************
#  アップロードデータ作成  https://qiita.com/5zm/items/ac8c9d1d74d012e682b4
#***********************************************************************************

UPLOAD_DIR = "./static/result/"
label_names = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]

@application.route('/upload', methods=['POST'])
def upload():

    #アップロードファイルの読込
    if 'uploadFile' not in request.files:
        return render_template('Upload.html', file_url='', result = 'uploadFile is required')

    file = request.files['uploadFile']
    fileName = file.filename

    if '' == fileName:
        return render_template('Upload.html', file_url='', result = 'uploadFile is required')

    #アップロードファイルを保存
    try:
        #saveFileName = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + fileName
        saveFileName = fileName
        os_saveFileName = os.path.join(UPLOAD_DIR, saveFileName)  
        file.save(os_saveFileName)
        file_url = "result/" + saveFileName
    except:
        return render_template('Upload.html', file_url='', result = 'save  error')

    try:
        #モデルの読込
        model_loaded=tf.keras.models.load_model("./static/cifar10.h5")  
        #画像の読込とtensorflowが処理できるように変換
        img_path = os_saveFileName
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32, 32))
        x = tf.keras.preprocessing.image.array_to_img(img)
        x = np.expand_dims(x, axis=0) 

    except:
        return render_template('Upload.html', file_url=file_url, result = 'model_load error')

    #アップロードファイルを識別
    try:
        pred = model_loaded.predict_classes(x)      
        result = label_names[pred[0]]
    except:
        return render_template('Upload.html', file_url='', result = 'pred error')

    return render_template('Upload.html', file_url=file_url, status='ok',result = result)




#リクエストエンティティが大きすぎます
@application.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    print("werkzeug.exceptions.RequestEntityTooLarge")
    return 'result : file size is overed.'    
#************************************** 
#  サーバ起動  マルチスレッド指定 デフォルトはTrueの動きをするようだが。 https://qiita.com/5zm/items/251be97d2800bf67b1c6
#************************************** 
if __name__ == '__main__':
    application.debug = True # デバッグ
    application.run(host='0.0.0.0', port=8000, threaded=True)
