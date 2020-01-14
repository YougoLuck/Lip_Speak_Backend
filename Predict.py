import tensorflow as tf
import pickle
import numpy as np
import cv2


def converIntToLabels(utilDict, labels):
    index2char = utilDict['index2char']
    stringLabels = []
    for label in labels:
        string = ''

        for intLabel in label:
            if intLabel == -1:
                continue
            string = string + index2char[intLabel]
        stringLabels.append(string)
    return stringLabels


def readVideo(filename):

    cap = cv2.VideoCapture(filename)
    frames = []
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()
    cap.release()
    return frames


def readVideos(filenames):
    videos = []
    videoLens = []
    for filename in filenames:
        video = readVideo(filename)
        videoLens.append(len(video))
        videos.append(video)

    videos = np.array(videos, dtype=np.uint8)
    return videos, videoLens





def predict(videoPaths, utilDict):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('put your model graph here')
        saver.restore(sess, tf.train.latest_checkpoint('put your model here'))
        graph = tf.get_default_graph()
        videoInput = graph.get_tensor_by_name('video_input:0')
        videoLengths = graph.get_tensor_by_name('video_length:0')
        channel_keep_prob = graph.get_tensor_by_name('channel_keep_prob:0')
        _y = graph.get_tensor_by_name('out_decoded:0')
        videos, videoLens = readVideos(videoPaths)
        y = sess.run([_y], feed_dict={videoInput: videos,
                                      videoLengths: videoLens,
                                      channel_keep_prob: 1.})
        result = converIntToLabels(utilDict, y[0])
    return result

