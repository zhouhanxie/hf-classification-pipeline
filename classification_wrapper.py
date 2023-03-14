import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import joblib
import os

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    """https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class ClassificationWrapper:
    
    def __init__(
        self, 
        model_name_or_path,
        device='cuda:0', 
        max_length=148
    ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model.to(device)
        try:
            self.label_encoder = joblib.load(os.path.join(model_name_or_path, 'label_encoder.joblib'))
        except:
            self.label_encoder = None
            print('could not found label mapping, returning numerical predictions')
        
    def _extract(self, raw_inp):
        
        inp = raw_inp
            
        with torch.no_grad():
            encoded_src = self.tokenizer(
                inp, 
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            ).to(self.device)
            output = torch.argmax(self.model(**encoded_src).logits, dim=-1).cpu().numpy()
            
        if self.label_encoder is not None:
            output_dict = [{'input':k, 'predicted':self.label_encoder.inverse_transform([v])[0]} 
                           for k,v in dict(zip(raw_inp, output)).items()]
        else:
            output_dict = [{'input':k, 'predicted':v} for k,v in dict(zip(raw_inp, output)).items()]
        
        return output_dict
    
    def __call__(self, sentence, batch_size=4, disable_tqdm=False, on_error='raise'):
        """
        sentence: Union[List, String]
        batch_size: int
        distable_tqdm: bool
        on_error: string in {'raise', 'echo', 'nothing'}
        """
        assert on_error in {'raise', 'echo', 'nothing'}
        if type(sentence) not in (list, str):
            raise TypeError
        if type(sentence) == str:
            raw_inp = [sentence]
        else:
            raw_inp = sentence
            
        batches = list(chunks(raw_inp, batch_size))
        outputs = []
        
        for batch in tqdm(batches, disable=disable_tqdm):
            try:
                outputs += self._extract(batch)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if on_error == 'nothing':
                    pass
                elif on_error == 'echo':
                    print('---')
                    print('T5MovieExtractor: oops broken batch, sorry...')
                    print(batch)
                    print(e)
                    print('---')
                raise e
            
        if type(sentence) == str:
            return outputs[0]
        return outputs