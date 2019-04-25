import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from top_down_bottom_up.question_embeding import QuestionEmbeding
from top_down_bottom_up.question_embeding import build_question_encoding_module


class qa_baseline(nn.Module):
    def __init__(self, **kwargs):
        super(qa_baseline, self).__init__()
        emb_config = kwargs['question_embedding'][0]
        key, par = "default_que_embed", emb_config['par']
        self.question_emb = build_question_encoding_module(key, par, num_vocab=kwargs['num_vocab'])
        self.caption_emb = build_question_encoding_module(key, par, num_vocab=kwargs['num_vocab'])
        self.linear = nn.Linear(in_features=2 * par['LSTM_hidden_size'], out_features=kwargs['out_dim'])

    def forward(self, input_question_variable, input_captions_variable, **kwargs):
        # print("forward question: ", input_question_variable.shape, "captions: ", input_captions_variable.shape)
        question_feat = self.question_emb(input_question_variable)
        caption_feat = self.caption_emb(input_captions_variable)
        feat = torch.cat((question_feat, caption_feat), dim=1)
        out = self.linear(feat)
        return out


