import connexion
from typing import Dict
from typing import Tuple
from typing import Union

from timestep.api.openai.v1.models.delete_model_response import DeleteModelResponse  # noqa: E501
from timestep.api.openai.v1.models.list_models_response import ListModelsResponse  # noqa: E501
from timestep.api.openai.v1.models.model import Model  # noqa: E501
from timestep.api.openai.v1 import util

from openai.types.model import Model

from timestep.database import InstanceStoreSingleton
from llama_cpp import Llama

instance_store = InstanceStoreSingleton()

def delete_model(model):  # noqa: E501
    """Delete a fine-tuned model. You must have the Owner role in your organization to delete a model.

     # noqa: E501

    :param model: The model to delete
    :type model: str

    :rtype: Union[DeleteModelResponse, Tuple[DeleteModelResponse, int], Tuple[DeleteModelResponse, int, Dict[str, str]]
    """
    raise NotImplementedError


def list_models():  # noqa: E501
    """Lists the currently available models, and provides basic information about each one such as the owner and availability.

     # noqa: E501


    :rtype: Union[ListModelsResponse, Tuple[ListModelsResponse, int], Tuple[ListModelsResponse, int, Dict[str, str]]
    """
    raise NotImplementedError


def retrieve_model(model_id: str):
    """Retrieves a model instance, providing basic information about the model such as the owner and permissioning.

     # noqa: E501

    :param model: The ID of the model to use for this request
    :type model: str

    :rtype: Union[Model, Tuple[Model, int], Tuple[Model, int, Dict[str, str]]
    """

    model: Llama = instance_store._shared_instance_state["models"][model_id]

    return {
        "chat_format": model.chat_format,
        "model_path": model.model_path,
        "n_ctx": model.n_ctx(),
        "lora_base": model.lora_base,
        "lora_path": model.lora_path,
        "lora_scale": model.lora_scale,
    }