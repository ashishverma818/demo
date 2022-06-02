from abc import ABC
import logging
import ast

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class NLGHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    Adapted from https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers
    """
    def __init__(self):
        super(NLGHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        # Read model serialize/pt file
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        return data


    def inference(self, inputs):
        """
        Predict the answer for a question in the text using a trained transformer model.
        """
        self.model.eval()
        inputs = self.tokenizer.encode("WebNLG:{} </s>".format(inputs), return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs  = self.model.generate(inputs,max_length= 3000)
        gen_text = self.tokenizer.decode(outputs[0]).replace('<pad>','').replace('</s>','')
        logger.info("Model predicted: '%s'", gen_text)

        return [gen_text]

    def postprocess(self, inference_output):
        return inference_output

    def handle(self, data, context):
        try:
            if data is None:
                 return None

            for inp in data:
                inp = self.preprocess(inp)
                inp = self.inference(inp)
                inp = self.postprocess(inp)
                
            return inp
        except Exception as e:
            raise e


