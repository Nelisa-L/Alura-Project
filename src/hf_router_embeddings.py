"""
hf_router_embeddings.py
------------------------
Clase de embeddings compatible con LangChain que llama directamente al
endpoint ACTUAL de la API de inferencia de HuggingFace.

HuggingFace descontinuo el dominio antiguo (api-inference.huggingface.co)
en favor de router.huggingface.co. La integracion incluida en
langchain_community (HuggingFaceInferenceAPIEmbeddings) todavia apunta
al dominio viejo, por lo que implementamos aqui una version minima y
directa, sin depender de esa clase desactualizada.

No requiere ningun modelo pesado en memoria local: solo hace peticiones
HTTP a la API de HuggingFace, ideal para entornos con RAM limitada
(como el tier gratuito de Render).
"""
import requests
from langchain_core.embeddings import Embeddings

ROUTER_BASE_URL = "https://router.huggingface.co/hf-inference/models"


class HFRouterEmbeddings(Embeddings):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.url = f"{ROUTER_BASE_URL}/{model_name}/pipeline/feature-extraction"

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Error llamando a la API de HuggingFace ({response.status_code}): "
                f"{response.text[:300]}"
            )
        data = response.json()

        # La API puede devolver:
        # - un vector ya promediado por texto (lista de floats), o
        # - una matriz token-a-token (lista de listas de floats) que hay
        #   que promediar (mean pooling) para obtener un solo vector.
        results = []
        for item in data:
            if isinstance(item[0], (float, int)):
                # Ya es un vector plano
                results.append(item)
            else:
                # Es una matriz [tokens x dim]; promediamos (mean pooling)
                num_tokens = len(item)
                dim = len(item[0])
                avg = [sum(tok[d] for tok in item) / num_tokens for d in range(dim)]
                results.append(avg)
        return results

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Procesamos en lotes pequenos para no exceder limites de la API
        batch_size = 16
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._call_api([text])[0]
