from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoModel, AutoTokenizer
from transformers import LukeTokenizer, LukeModel


def get_baseline_model(model_name):

    if model_name == 'bert-base':
        name_str = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(name_str)
        model = BertModel.from_pretrained(name_str)
        

    elif model_name == 'bert-large':
        name_str = 'bert-large-uncased'
        tokenizer = BertTokenizer.from_pretrained(name_str)
        model = BertModel.from_pretrained(name_str)
        

    elif model_name == 'roberta-base':
        name_str = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(name_str)
        model = RobertaModel.from_pretrained(name_str)
        
    elif model_name == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('roberta-large')
           
    elif model_name == 'spanbert-base':
        tokenizer = AutoTokenizer.from_pretrained('SpanBERT/spanbert-base-cased')
        model = AutoModel.from_pretrained('SpanBERT/spanbert-base-cased')
        
    elif model_name == 'spanbert-large':
        tokenizer = AutoTokenizer.from_pretrained('SpanBERT/spanbert-large-cased')
        model = AutoModel.from_pretrained('SpanBERT/spanbert-large-cased')
        
    elif model_name == 'luke-base':
        tokenizer = LukeTokenizer.from_pretrained('studio-ousia/luke-base')
        model = LukeModel.from_pretrained('studio-ousia/luke-base')
        

    elif model_name == 'luke-large':
        tokenizer = LukeTokenizer.from_pretrained('studio-ousia/luke-large')
        model = LukeModel.from_pretrained('studio-ousia/luke-large')
        

    elif model_name == 'simcse-bert-base':
        name_str = 'princeton-nlp/unsup-simcse-bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        model = AutoModel.from_pretrained(name_str)
        
    elif model_name == 'simcse-bert-large':
        name_str = 'princeton-nlp/unsup-simcse-bert-large-uncased'
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        model = AutoModel.from_pretrained(name_str)
        

    elif model_name == 'simcse-roberta-base':
        name_str = 'princeton-nlp/unsup-simcse-roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        model = AutoModel.from_pretrained(name_str)
        
    elif model_name == 'simcse-roberta-large':
        name_str = 'princeton-nlp/unsup-simcse-roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        model = AutoModel.from_pretrained(name_str)
        

    else:
        raise NotImplementedError

    return model, tokenizer
