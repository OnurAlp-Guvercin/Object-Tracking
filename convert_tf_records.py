def create_tf_records(annotation_file,images_dir,tfrecords_dir):
    import os
    import json
    import pprint
    import tensorflow as tf
    import matplotlib.pyplot as plt

    with open(annotation_file, "r") as f:
        annotations = json.load(f)["annotations"]


    num_samples = len(annotations)
    num_tfrecords = len(annotations) // num_samples
    if len(annotations) % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)  # creating TFRecords output folder

    def image_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
        )


    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


    def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def float_feature_list(value):
        """Returns a list of float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    def create_example(image, path, example):
        feature = {
            "image": image_feature(image),
            "path": bytes_feature(path),
            "area": float_feature(example["area"]),
            "bbox": float_feature_list(example["bbox"]),
            "category_id": int64_feature(example["category_id"]),
            "id": int64_feature(example["id"]),
            "image_id": int64_feature(example["image_id"]),
            "track_id": int64_feature(example["track_id"]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    for tfrec_num in range(num_tfrecords):
        samples = annotations[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]

        with tf.io.TFRecordWriter(
            tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        ) as writer:
            for sample in samples:
                image_path = f"{images_dir}/{(sample['image_id']-1):04d}.jpg"
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, image_path, sample)
                writer.write(example.SerializeToString())
    

create_tf_records(annotation_file = 'validation_images.json',images_dir = "validation_images",tfrecords_dir = "tf_records_validation")