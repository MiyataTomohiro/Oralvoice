import io
import os
import sys
import platform
import datetime
import time


from flask import *

import json

import sqlite3

import werkzeug
import urllib.request

import pickle

import numpy as np

#画面に表示しようとした時
#from PIL import Image  #Python Image Libraryインストールは pip install Pillow

#**************************************
# 共通処理：SQLiteの戻り値が単純な配列なのでそれを辞書型にする定義
#**************************************
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d
#**************************************
# ログ出力  if os.path.exists('err.log'):  else:
#**************************************
def log(msg):
    errlog = open('./LogFiles/err.log','a')
    errlog.write( msg + '\n')
    errlog.close()