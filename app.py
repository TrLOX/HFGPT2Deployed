import torch
import argparse
from multiprocessing import cpu_count

import ray
from ray import serve

from fastapi import FastAPI
from transformers import pipeline


app = FastAPI()
serve_handle = None
args=argparse.Namespace()
#use_gpu = torch.cuda.is_available()
use_gpu = false
args.device = torch.device("cuda" if use_gpu else "cpu")
num_rep = cpu_count()
if use_gpu: num_rep = torch.cuda.device_count()

@app.on_event("startup")
async def startup_event():
	ray.init(address="auto")
	client = serve.start()

	class GPT2:
		def __init__(self):
			self.nlp_model = pipeline('text-generation', model='gpt2')

		def __call__(self, request):
			return self.nlp_model(request._data, max_length=50)

	backend_config = serve.BackendConfig(num_replicas=num_rep)
	client.create_backend("gpt-2", GPT2, args, config=backend_config)
	client.create_endpoint("generate", backend="gpt-2")

	global serve_handle
	serve_handle = client.get_handle("generate")


@app.get('/generate')
async def generate(query: str):
	return await serve_handle.remote(query)
