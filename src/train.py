import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier, optimization, tokenization

import argparse
import metrics

# path to cased multilingual model
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

tf.logging.set_verbosity(tf.logging.INFO)


def load_dataset(data_dir):
    train_df = pd.read_csv(f'{data_dir}/train.tsv', sep='\t')
    test_df = pd.read_csv(f'{data_dir}/dev.tsv', sep='\t')

    return train_df, test_df


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=True)
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)


# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels)
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


# define finetuning hyperparams
# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)


def convert_dataframe_to_input_example_fn(data_column, label_column):
    return lambda x: bert.run_classifier.InputExample(guid=None,
                                                      # Globally unique ID for bookkeeping, unused in this example
                                                      text_a=x[data_column],
                                                      text_b=None,
                                                      label=x[label_column])


def main(args):
    train_df, test_df = load_dataset(args.data_dir)

    label_list = list(range(args.num_classes))

    train_InputExamples = train_df.apply(convert_dataframe_to_input_example_fn(args.data_column, args.label_column),
                                         axis=1)

    test_InputExamples = test_df.apply(convert_dataframe_to_input_example_fn(args.data_column, args.label_column),
                                       axis=1)

    tokenizer = create_tokenizer_from_hub_module()

    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples,
                                                                      label_list,
                                                                      args.max_seq_length,
                                                                      tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples,
                                                                     label_list,
                                                                     args.max_seq_length,
                                                                     tokenizer)

    num_train_steps = int(len(train_features) / args.batch_size * args.num_train_epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    run_config = tf.estimator.RunConfig(
        model_dir=args.output_dir,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": args.batch_size})

    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=args.max_seq_length,
        is_training=True,
        drop_remainder=False)

    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=args.max_seq_length,
        is_training=False,
        drop_remainder=False)

    tf.logging.info('Beginning finetuning ...')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    tf.logging.info(f'Finished finetuning!')
    tf.logging.info(f'Time takem: {datetime.now() - current_time}')

    # eval step
    predictions = estimator.predict(input_fn=test_input_fn)
    test_df['pred'] = [prediction['labels'] for prediction in predictions]

    tf.logging.info('Computing evaluation metrics\n')
    eval_metrics = metrics.compute_eval_metrics(test_df)

    metrics.print_metrics(eval_metrics)

    test_df.to_csv(f'{args.output_dir}/eval_results.tsv',
                   sep='\t',
                   index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--data_column', type=str, default='text')

    parser.add_argument('--label_column', type=str, default='target')
    parser.add_argument('--num_classes', required=True, type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_train_epochs', type=float, default=3)

    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    args = parser.parse_args()

    main(args)
