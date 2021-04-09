import ray
from ray import serve

from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

serve_handle = None


@app.on_event("startup")  # Code to be run when the server starts.
async def startup_event():
    ray.init(address="auto")  # Connect to the running Ray cluster.
    serve.start(http_host=None)  # Start the Ray Serve instance.

    # Define a callable class to use for our Ray Serve backend.
    class GPT2:
        def __init__(self):
            print("This actor is allowed to use GPUs {}.".format(ray.get_gpu_ids()))
            self.nlp_model = pipeline("text-generation", model="gpt2m",device=0)

        async def __call__(self, request):
            return self.nlp_model(await request.body(),max_length=1000, early_stopping=True, top_p=.90, top_k=50 , do_sample=True , num_beams=6, 
    no_repeat_ngram_size=2)

    # Set up a Ray Serve backend with the desired number of replicas.
    backend_config = serve.BackendConfig(num_replicas=1)
    serve.create_backend("gpt-2", GPT2, config=backend_config,ray_actor_options={"num_gpus": 1})
    serve.create_endpoint("generate", backend="gpt-2")

    # Get a handle to our Ray Serve endpoint so we can query it in Python.
    global serve_handle
    serve_handle = serve.get_handle("generate")


@app.get("/generate")
async def generate(query: str):
    return await serve_handle.remote(query)
