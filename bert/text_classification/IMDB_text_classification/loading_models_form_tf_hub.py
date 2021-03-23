# -*- encoding: utf-8 -*-
'''
@File    :   loading_models_form_tf_hub.py
@Time    :   2021/03/18 11:41:40
@Author  :   Liang Xiaoguang
@Contact :   hplxg@hotmail.com
--**--
--**--
'''
import tensorflow as tf
import tensorflow_hub as hub

from bert.text_classification.IMDB_text_classification.make_input_data import tfhub_handle_encoder, \
    tfhub_handle_preprocess, text_preprocessed


tf.get_logger().setLevel('ERROR')

bert_model = hub.KerasLayer(tfhub_handle_encoder)

bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')
