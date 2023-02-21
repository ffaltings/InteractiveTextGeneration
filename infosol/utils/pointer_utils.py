import sys
import torch

sys.path.append('/home/v-fefal/POINTER') 
from pytorch_transformers import BertForMaskedLM
from pytorch_transformers.tokenization_bert import BertTokenizer
from inference import convert_example_to_features, greedy_search, PregeneratedDataset
from util import MAX_TURN, PREVENT_FACTOR, PROMOTE_FACTOR, PREVENT_LIST, REDUCE_LIST, STOP_LIST, boolean_string

class POINTERArgs:
    
    def __init__(self, bert_model, do_lower_case=False, noi_decay=1, reduce_decay=1, prevent=True, 
                reduce_stop=True, lessrepeat=True, max_seq_length=256, no_ins_at_first=False, verbose=0):
        self.bert_model = bert_model
        self.do_lower_case = do_lower_case
        self.noi_decay = noi_decay
        self.reduce_decay = reduce_decay
        self.prevent = prevent
        self.reduce_stop = reduce_stop
        self.lessrepeat = lessrepeat
        self.max_seq_length = max_seq_length
        self.no_ins_at_first = no_ins_at_first
        self.verbose = verbose

class POINTERWrapper:
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        self.model = BertForMaskedLM.from_pretrained(args.bert_model)
        self.model.eval()
        
        prevent = [ self.tokenizer.vocab.get(x) for x in PREVENT_LIST] if args.prevent else None
        if args.reduce_stop:
            # import pdb; pdb.set_trace()
            reduce_l = REDUCE_LIST |  STOP_LIST
        reduce = None
        if args.prevent:
            reduce = [ self.tokenizer.vocab.get(x) for x in reduce_l]  
            reduce = [s for s in reduce if s]
        self.prevent = prevent
        self.reduce = reduce
 
    def prep_input(self, inp_tokens, device):
#         canvas = [c.strip().lstrip() for c in canvas]
        features = convert_example_to_features(inp_tokens, self.tokenizer, self.args.max_seq_length, 
                                               no_ins_at_first = self.args.no_ins_at_first, id=0, tokenizing=True)
        out = (features.input_ids, features.input_mask, features.segment_ids, features.lm_label_ids, features.no_ins)
        out = tuple(torch.tensor(o.reshape(1,-1)).long().to(device) for o in out)
        return out
    
    def generate_(self, input_ids, segment_ids, input_mask, no_ins):
        sep_tok = self.tokenizer.vocab['[SEP]']
        cls_tok = self.tokenizer.vocab['[CLS]']
        pad_tok = self.tokenizer.vocab['[PAD]']
    
        predict_ids = greedy_search(self.model, input_ids, segment_ids, input_mask, no_ins = no_ins, args=self.args,
                                   tokenizer=self.tokenizer, prevent=self.prevent, reduce=self.reduce)
        output =  " ".join([str(self.tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) for x in predict_ids[0].detach().cpu().numpy() if x!=sep_tok and x != pad_tok and x != cls_tok]) 
        output = output.replace(" ##", "")
        return output
    
    def generate(self, inp_tokens, device=torch.device('cpu')):
        input_ids, input_mask, segment_ids, lm_label_ids, no_ins = self.prep_input(inp_tokens, device)
        return self.generate_(input_ids, segment_ids, input_mask, no_ins)
