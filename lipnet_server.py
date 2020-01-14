from flask import Flask, jsonify, request
import hashlib
from Predict import predict
import pickle
import os

app = Flask(__name__)

trainLabel = {}

SERVER_ADDRESS = '192.168.123.169'
UPLOAD_DIR = 'data'
TEST_UPLOAD_DIR = 'tem_data'

DATA_SAVE_PATH = os.path.join(UPLOAD_DIR, 'train_data')
DATA_LOW_SAVE_PATH = os.path.join(UPLOAD_DIR, 'train_data_low')
TEM_SAVE_PATH = os.path.join(TEST_UPLOAD_DIR, 'train_data')
TEM_LOW_SAVE_PATH = os.path.join(TEST_UPLOAD_DIR, 'train_data_low')


def loadUtilDict(filePath):
    with open(filePath, 'rb') as f:
        storeDict = pickle.load(f)
        return storeDict


utilDict = loadUtilDict('utilDict.pkl')


@app.route('/checkServer', methods=['POST'])
def check_server():
    return jsonify({'code': 0,
                    'msg': 'Server Connected'})


@app.route('/uploadTrainData', methods=['POST'])
def uploadTrainData():

    check_state = checkResponseFile(request.files)
    if check_state != 1:
        return check_state
    file = request.files['file']

    param = request.form
    if 'label' in param.keys() and '' != param['label']:
        label = param['label']
    else:
        return jsonify({'code': -3,
                        'msg': 'Label must not be empty'})
    fileData = file.read()
    fileName = getFileMD5(fileData)
    originSavePath = os.path.join(DATA_SAVE_PATH, fileName)
    resizeOutputPath = os.path.join(DATA_LOW_SAVE_PATH, fileName)

    if os.path.exists(originSavePath):
        return jsonify({'code': -4,
                        'msg': 'This file already uploaded'})

    writeFile(originSavePath, fileData)

    if resizeMovie(originSavePath, resizeOutputPath) != 0:
        removeFile(originSavePath)
        removeFile(resizeOutputPath)
        return jsonify({'code': -5,
                        'msg': 'Resize video failed'})

    writeLabel(fileName, label)
    return jsonify({'code': 0,
                    'msg': 'Upload done'})


@app.route('/uploadTestTemData', methods=['POST'])
def uploadTestTemData():
    check_state = checkResponseFile(request.files)
    if check_state != 1:
        return check_state
    file = request.files['file']

    fileData = file.read()
    fileName = getFileMD5(fileData)

    originSavePath = os.path.join(TEM_SAVE_PATH, fileName)
    resizeOutputPath = os.path.join(TEM_LOW_SAVE_PATH, fileName)

    if not os.path.exists(originSavePath):
        writeFile(originSavePath, fileData)
        if resizeMovie(originSavePath, resizeOutputPath) != 0:
            removeFile(originSavePath)
            removeFile(resizeOutputPath)
            return jsonify({'code': -5,
                            'msg': 'Resize video failed'})

    result = predict([resizeOutputPath], utilDict)

    return jsonify({'code': 0,
                    'result': result,
                    'msg': 'Predict done'})


def writeLabel(file, label):
    f = open(os.path.join(UPLOAD_DIR, 'trainDATA_label.txt'), 'a', encoding='utf-8')
    line = file + ',' + label + '\n'
    f.write(line)
    f.close()


def writeFile(path, data):
    f = open(path, "wb")
    f.write(data)
    f.close()


def removeFile(path):
    if os.path.exists(path):
        os.remove(path)


def checkResponseFile(files):
    if 'file' not in files.keys() or not files['file']:
        return jsonify({'code': -1,
                        'msg': 'File must not be empty'})
    file = files['file']
    if file.content_type != 'mp4/video':
        return jsonify({'code': -2,
                        'msg': 'Content-type error'})

    if '' == file.filename:
        return jsonify({'code': -3,
                        'msg': 'File name must not be empty'})
    return 1


def resizeMovie(originPath, outputPath):
    print('ffmpeg -i {} -vf scale=100:66 {}'.format(originPath, outputPath))
    return os.system('ffmpeg -i {} -vf scale=100:66 {}'.format(originPath, outputPath))


def getFileMD5(fileData):
    return hashlib.md5(fileData).hexdigest() + '.mp4'


def serverRun():
    file_dirs = [DATA_SAVE_PATH, DATA_LOW_SAVE_PATH, TEM_SAVE_PATH, TEM_LOW_SAVE_PATH]
    for dir_name in file_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    app.run(host=SERVER_ADDRESS, port=55080)


if __name__ == '__main__':
    serverRun()
