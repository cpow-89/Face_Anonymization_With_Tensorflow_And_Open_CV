import tensorflow as tf
import os
import helper
from collections import namedtuple


TRAIN_CONFIG = namedtuple("TRAIN_CONFIG", ["EPOCHS", "BATCH_SIZE", "DROPOUT_KEEP_PROB", "WRITE_SUMMARY", "CHECKPOINT_DIR", "SUMMARY_DIR", "TRAIN_DATA_PATH"])


def _save_session(sess, checkpoint_dir):
    saver = tf.train.Saver()
    current_date_time = helper.get_current_date_time()
    current_date_time = current_date_time.replace(" ", "__").replace("/", "_").replace(":", "_")
    helper.mkdir(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, "ckpt_{}.ckpt".format(current_date_time)))


def _print_eval_results(loss, epoch):
    print("Evaluation at Epoch: {} - Loss: {}".format(epoch, loss))


def _run_train(sess, model, x_train, y_train, x_eval, y_eval, train_config):
    
    for epoch in range(train_config.EPOCHS):
        # train
        for batch_x, batch_y in helper.get_batch(x_train, y_train, step=train_config.BATCH_SIZE):
            model.train(sess, batch_x, batch_y, dropout_keep_prob=train_config.DROPOUT_KEEP_PROB)
        # eval
        loss = model.evaluate_loss(sess, x_eval, y_eval, dropout_keep_prob=1.0)
        _print_eval_results(loss, epoch)
        

def _run_train_with_summary(sess, model, x_train, y_train, x_eval, y_eval, train_config):
    saver = tf.train.Saver()
    helper.mkdir(train_config.SUMMARY_DIR)
    summary_writer = tf.summary.FileWriter(logdir=train_config.SUMMARY_DIR)
    summary_writer.add_graph(sess.graph)
    for epoch in range(train_config.EPOCHS):
        # train
        for batch_x, batch_y in helper.get_batch(x_train, y_train, step=train_config.BATCH_SIZE):
            summary, _ = model.train(sess, batch_x, batch_y, dropout_keep_prob=train_config.DROPOUT_KEEP_PROB)
            summary_writer.add_summary(summary)
        # eval
        summary, loss = model.evaluate_loss(sess, batch_x, batch_y, dropout_keep_prob=1.0)
        summary_writer.add_summary(summary)
        _print_eval_results(loss, epoch)


def train(model, x_train, y_train, x_eval, y_eval, train_config):
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train
        if train_config.WRITE_SUMMARY:
            _run_train_with_summary(sess, model, x_train, y_train, x_eval, y_eval, train_config)
        else:
            _run_train(sess, model, x_train, y_train, x_eval, y_eval, train_config)
        # save
        _save_session(sess, train_config.CHECKPOINT_DIR)


def facial_key_point_detection(image, face_key_point_detector, sess_restore_dir, scale_factor, min_neighbors):
    saver = tf.train.Saver()
    face_key_points = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(sess_restore_dir))
        face_key_points.append(face_key_point_detector.detect(sess, image, scale_factor=scale_factor,
                                                              min_neighbors=min_neighbors))
    return face_key_points


def facial_anonymisation(image, overlay_image, face_anonymizer, sess_restore_dir, scale_factor, min_neighbors):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(sess_restore_dir))
        anonymised_image = face_anonymizer.anonymize(sess, image, overlay_image, scale_factor, min_neighbors)
    return anonymised_image

