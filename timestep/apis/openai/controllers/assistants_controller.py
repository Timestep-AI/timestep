import time
import uuid
import connexion
from typing import Dict, List
from typing import Tuple
from typing import Union

from openai.types.beta.assistant import Assistant
from openai.types.beta.thread import ToolResourcesCodeInterpreter, ToolResourcesFileSearch, Thread
from openai.types.beta.threads.message import Message, MessageContent
from openai.types.beta.threads.run import Run, RunStatus
from openai.types.beta.threads.runs import RunStep
from openai.types.beta.threads.text import Text
from openai.types.beta.threads.text_content_block import TextContentBlock

from prefect_sqlalchemy import AsyncDriver, SqlAlchemyConnector, ConnectionComponents, SyncDriver

from timestep.apis.openai.models.assistant_object import AssistantObject  # noqa: E501
from timestep.apis.openai.models.create_assistant_request import CreateAssistantRequest  # noqa: E501
from timestep.apis.openai.models.create_message_request import CreateMessageRequest  # noqa: E501
from timestep.apis.openai.models.create_run_request import CreateRunRequest  # noqa: E501
from timestep.apis.openai.models.create_thread_and_run_request import CreateThreadAndRunRequest  # noqa: E501
from timestep.apis.openai.models.create_thread_request import CreateThreadRequest  # noqa: E501
from timestep.apis.openai.models.delete_assistant_response import DeleteAssistantResponse  # noqa: E501
from timestep.apis.openai.models.delete_message_response import DeleteMessageResponse  # noqa: E501
from timestep.apis.openai.models.delete_thread_response import DeleteThreadResponse  # noqa: E501
from timestep.apis.openai.models.list_assistants_response import ListAssistantsResponse  # noqa: E501
from timestep.apis.openai.models.list_messages_response import ListMessagesResponse  # noqa: E501
from timestep.apis.openai.models.list_run_steps_response import ListRunStepsResponse  # noqa: E501
from timestep.apis.openai.models.list_runs_response import ListRunsResponse  # noqa: E501
from timestep.apis.openai.models.message_object import MessageObject  # noqa: E501
from timestep.apis.openai.models.modify_assistant_request import ModifyAssistantRequest  # noqa: E501
from timestep.apis.openai.models.modify_message_request import ModifyMessageRequest  # noqa: E501
from timestep.apis.openai.models.modify_run_request import ModifyRunRequest  # noqa: E501
from timestep.apis.openai.models.modify_thread_request import ModifyThreadRequest  # noqa: E501
from timestep.apis.openai.models.run_object import RunObject  # noqa: E501
from timestep.apis.openai.models.run_step_object import RunStepObject  # noqa: E501
from timestep.apis.openai.models.submit_tool_outputs_run_request import SubmitToolOutputsRunRequest  # noqa: E501
from timestep.apis.openai.models.thread_object import ThreadObject  # noqa: E501
from timestep.apis.openai import util
from timestep.database import borg


def cancel_run(thread_id, run_id):  # noqa: E501
    """Cancels a run that is &#x60;in_progress&#x60;.

     # noqa: E501

    :param thread_id: The ID of the thread to which this run belongs.
    :type thread_id: str
    :param run_id: The ID of the run to cancel.
    :type run_id: str

    :rtype: Union[RunObject, Tuple[RunObject, int], Tuple[RunObject, int, Dict[str, str]]
    """
    return 'do some magic!'


def create_assistant(body):  # noqa: E501
    """Create an assistant with a model and instructions.

     # noqa: E501

    :param create_assistant_request: 
    :type create_assistant_request: dict | bytes

    :rtype: Union[AssistantObject, Tuple[AssistantObject, int], Tuple[AssistantObject, int, Dict[str, str]]
    """
    # if connexion.request.is_json:
    #     create_assistant_request = CreateAssistantRequest.from_dict(connexion.request.get_json())  # noqa: E501

    # return 'do some magic!'
    # kwargs:  {
    #     'body': {
    #         'model': 'gpt-4-1106-preview',
    #         'instructions': 'You are a personal math tutor. Answer questions briefly, in a sentence or less.',
    #         'name': 'Math Tutor'},
    #         'user': 'user_id',
    #         'token_info': {'uid': 'user_id'}
    #     }

    # assistant = Assistant(**body)
    assistant = Assistant(
        id=str(uuid.uuid4()),
        created_at=int(time.time()),
        description=body.get("description"),
        instructions=body.get("instructions"),
        model=body.get("model"),
        name=body.get("name"),
        object="assistant",
        tools=body.get("tools", [])
    )
    print('assistant: ', assistant)

    borg._shared_borg_state["assistants"][assistant.id] = assistant

    return assistant.model_dump(mode="json")


def create_message(body, token_info, thread_id, user):
    """Create a message.

     # noqa: E501

    :param thread_id: The ID of the [thread](/docs/api-reference/threads) to create a message for.
    :type thread_id: str
    :param create_message_request: 
    :type create_message_request: dict | bytes

    :rtype: Union[MessageObject, Tuple[MessageObject, int], Tuple[MessageObject, int, Dict[str, str]]
    """
    # print('args: ', args)
    # print('kwargs: ', kwargs)
    # {'thread_id': '6d2aa996-fea2-4fde-98a3-ec176a11b4e5',
    #  'body': {'content': 'I need to solve the equation `3x + 11 = 14`. Can you help me?', 'role': 'user'},
    #  'user': 'user_id',
    #  'token_info': {'uid': 'user_id'}}

    content: List[MessageContent] = []

    if body.get("content"):
        content.append(TextContentBlock(
            text=Text(
                annotations=[],
                value=body.get("content"),
            ),
            type="text",
        ))

    message = Message(
        id=str(uuid.uuid4()),
        content=content,
        created_at=int(time.time()),
        object="thread.message",
        role=body.get("role"),
        thread_id=thread_id,
        status="incomplete",
    )

    borg._shared_borg_state["messages"][message.id] = message

    return message.model_dump(mode="json")


def create_run(body, token_info, user, thread_id):
    """Create a run.

     # noqa: E501

    :param thread_id: The ID of the thread to run.
    :type thread_id: str
    :param create_run_request: 
    :type create_run_request: dict | bytes

    :rtype: Union[RunObject, Tuple[RunObject, int], Tuple[RunObject, int, Dict[str, str]]
    """
    # if connexion.request.is_json:
    #     create_run_request = CreateRunRequest.from_dict(connexion.request.get_json())  # noqa: E501
    # return 'do some magic!'

    # print('args: ', args)
    # print('kwargs: ', kwargs)

    # print('assistant_id, token_info, thread_id, user: ', assistant_id, token_info, thread_id, user)

    print('body: ', body)

    assistant_id = body.get("assistant_id")

    assistant: Assistant = borg._shared_borg_state["assistants"].get(assistant_id)
    thread: Thread = borg._shared_borg_state["threads"].get(thread_id)

    run = Run(
        id=str(uuid.uuid4()),
        assistant_id=assistant.id,
        created_at=int(time.time()),
        instructions=assistant.instructions,
        model=assistant.model,
        object="thread.run",
        parallel_tool_calls=False,
        status="queued",
        thread_id=thread.id,
        tools=assistant.tools,
    )

    borg._shared_borg_state["runs"][run.id] = run

    return run.model_dump(mode="json")


def create_thread(body, token_info, user):  # noqa: E501
    """Create a thread.

     # noqa: E501

    :param create_thread_request: 
    :type create_thread_request: dict | bytes

    :rtype: Union[ThreadObject, Tuple[ThreadObject, int], Tuple[ThreadObject, int, Dict[str, str]]
    """
    print('body: ', body)
    print('token_info: ', token_info)
    print('user: ', user)

    thread = Thread(
        id=str(uuid.uuid4()),
        created_at=int(time.time()),
        object="thread",
        # tool_resources
    )

    borg._shared_borg_state["threads"][thread.id] = thread

    return thread.model_dump(mode="json")


def create_thread_and_run(create_thread_and_run_request):  # noqa: E501
    """Create a thread and run it in one request.

     # noqa: E501

    :param create_thread_and_run_request: 
    :type create_thread_and_run_request: dict | bytes

    :rtype: Union[RunObject, Tuple[RunObject, int], Tuple[RunObject, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        create_thread_and_run_request = CreateThreadAndRunRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def delete_assistant(assistant_id):  # noqa: E501
    """Delete an assistant.

     # noqa: E501

    :param assistant_id: The ID of the assistant to delete.
    :type assistant_id: str

    :rtype: Union[DeleteAssistantResponse, Tuple[DeleteAssistantResponse, int], Tuple[DeleteAssistantResponse, int, Dict[str, str]]
    """
    return 'do some magic!'


def delete_message(thread_id, message_id):  # noqa: E501
    """Deletes a message.

     # noqa: E501

    :param thread_id: The ID of the thread to which this message belongs.
    :type thread_id: str
    :param message_id: The ID of the message to delete.
    :type message_id: str

    :rtype: Union[DeleteMessageResponse, Tuple[DeleteMessageResponse, int], Tuple[DeleteMessageResponse, int, Dict[str, str]]
    """
    return 'do some magic!'


def delete_thread(thread_id):  # noqa: E501
    """Delete a thread.

     # noqa: E501

    :param thread_id: The ID of the thread to delete.
    :type thread_id: str

    :rtype: Union[DeleteThreadResponse, Tuple[DeleteThreadResponse, int], Tuple[DeleteThreadResponse, int, Dict[str, str]]
    """
    return 'do some magic!'


def get_assistant(assistant_id):  # noqa: E501
    """Retrieves an assistant.

     # noqa: E501

    :param assistant_id: The ID of the assistant to retrieve.
    :type assistant_id: str

    :rtype: Union[AssistantObject, Tuple[AssistantObject, int], Tuple[AssistantObject, int, Dict[str, str]]
    """
    return 'do some magic!'


def get_message(thread_id, message_id):  # noqa: E501
    """Retrieve a message.

     # noqa: E501

    :param thread_id: The ID of the [thread](/docs/api-reference/threads) to which this message belongs.
    :type thread_id: str
    :param message_id: The ID of the message to retrieve.
    :type message_id: str

    :rtype: Union[MessageObject, Tuple[MessageObject, int], Tuple[MessageObject, int, Dict[str, str]]
    """
    return 'do some magic!'


def get_run(run_id, thread_id, token_info, user):
    """Retrieves a run.

     # noqa: E501

    :param thread_id: The ID of the [thread](/docs/api-reference/threads) that was run.
    :type thread_id: str
    :param run_id: The ID of the run to retrieve.
    :type run_id: str

    :rtype: Union[RunObject, Tuple[RunObject, int], Tuple[RunObject, int, Dict[str, str]]
    """

    run: Run = borg._shared_borg_state["runs"][run_id]

    print('run: ', run)

    assert run.thread_id == thread_id, f"{run.thread_id} != {thread_id}"

    return run.model_dump(mode="json")


def get_run_step(thread_id, run_id, step_id):  # noqa: E501
    """Retrieves a run step.

     # noqa: E501

    :param thread_id: The ID of the thread to which the run and run step belongs.
    :type thread_id: str
    :param run_id: The ID of the run to which the run step belongs.
    :type run_id: str
    :param step_id: The ID of the run step to retrieve.
    :type step_id: str

    :rtype: Union[RunStepObject, Tuple[RunStepObject, int], Tuple[RunStepObject, int, Dict[str, str]]
    """
    return 'do some magic!'


def get_thread(thread_id):  # noqa: E501
    """Retrieves a thread.

     # noqa: E501

    :param thread_id: The ID of the thread to retrieve.
    :type thread_id: str

    :rtype: Union[ThreadObject, Tuple[ThreadObject, int], Tuple[ThreadObject, int, Dict[str, str]]
    """
    return 'do some magic!'


def list_assistants(limit=None, order=None, after=None, before=None):  # noqa: E501
    """Returns a list of assistants.

     # noqa: E501

    :param limit: A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20. 
    :type limit: int
    :param order: Sort order by the &#x60;created_at&#x60; timestamp of the objects. &#x60;asc&#x60; for ascending order and &#x60;desc&#x60; for descending order. 
    :type order: str
    :param after: A cursor for use in pagination. &#x60;after&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include after&#x3D;obj_foo in order to fetch the next page of the list. 
    :type after: str
    :param before: A cursor for use in pagination. &#x60;before&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include before&#x3D;obj_foo in order to fetch the previous page of the list. 
    :type before: str

    :rtype: Union[ListAssistantsResponse, Tuple[ListAssistantsResponse, int], Tuple[ListAssistantsResponse, int, Dict[str, str]]
    """
    return 'do some magic!'


def list_messages(thread_id, limit=None, order=None, after=None, before=None, run_id=None):  # noqa: E501
    """Returns a list of messages for a given thread.

     # noqa: E501

    :param thread_id: The ID of the [thread](/docs/api-reference/threads) the messages belong to.
    :type thread_id: str
    :param limit: A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20. 
    :type limit: int
    :param order: Sort order by the &#x60;created_at&#x60; timestamp of the objects. &#x60;asc&#x60; for ascending order and &#x60;desc&#x60; for descending order. 
    :type order: str
    :param after: A cursor for use in pagination. &#x60;after&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include after&#x3D;obj_foo in order to fetch the next page of the list. 
    :type after: str
    :param before: A cursor for use in pagination. &#x60;before&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include before&#x3D;obj_foo in order to fetch the previous page of the list. 
    :type before: str
    :param run_id: Filter messages by the run ID that generated them. 
    :type run_id: str

    :rtype: Union[ListMessagesResponse, Tuple[ListMessagesResponse, int], Tuple[ListMessagesResponse, int, Dict[str, str]]
    """
    return 'do some magic!'


def list_run_steps(thread_id, run_id, limit=None, order=None, after=None, before=None):  # noqa: E501
    """Returns a list of run steps belonging to a run.

     # noqa: E501

    :param thread_id: The ID of the thread the run and run steps belong to.
    :type thread_id: str
    :param run_id: The ID of the run the run steps belong to.
    :type run_id: str
    :param limit: A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20. 
    :type limit: int
    :param order: Sort order by the &#x60;created_at&#x60; timestamp of the objects. &#x60;asc&#x60; for ascending order and &#x60;desc&#x60; for descending order. 
    :type order: str
    :param after: A cursor for use in pagination. &#x60;after&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include after&#x3D;obj_foo in order to fetch the next page of the list. 
    :type after: str
    :param before: A cursor for use in pagination. &#x60;before&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include before&#x3D;obj_foo in order to fetch the previous page of the list. 
    :type before: str

    :rtype: Union[ListRunStepsResponse, Tuple[ListRunStepsResponse, int], Tuple[ListRunStepsResponse, int, Dict[str, str]]
    """
    return 'do some magic!'


def list_runs(thread_id, limit=None, order=None, after=None, before=None):  # noqa: E501
    """Returns a list of runs belonging to a thread.

     # noqa: E501

    :param thread_id: The ID of the thread the run belongs to.
    :type thread_id: str
    :param limit: A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20. 
    :type limit: int
    :param order: Sort order by the &#x60;created_at&#x60; timestamp of the objects. &#x60;asc&#x60; for ascending order and &#x60;desc&#x60; for descending order. 
    :type order: str
    :param after: A cursor for use in pagination. &#x60;after&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include after&#x3D;obj_foo in order to fetch the next page of the list. 
    :type after: str
    :param before: A cursor for use in pagination. &#x60;before&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include before&#x3D;obj_foo in order to fetch the previous page of the list. 
    :type before: str

    :rtype: Union[ListRunsResponse, Tuple[ListRunsResponse, int], Tuple[ListRunsResponse, int, Dict[str, str]]
    """
    return 'do some magic!'


def modify_assistant(assistant_id, modify_assistant_request):  # noqa: E501
    """Modifies an assistant.

     # noqa: E501

    :param assistant_id: The ID of the assistant to modify.
    :type assistant_id: str
    :param modify_assistant_request: 
    :type modify_assistant_request: dict | bytes

    :rtype: Union[AssistantObject, Tuple[AssistantObject, int], Tuple[AssistantObject, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        modify_assistant_request = ModifyAssistantRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def modify_message(thread_id, message_id, modify_message_request):  # noqa: E501
    """Modifies a message.

     # noqa: E501

    :param thread_id: The ID of the thread to which this message belongs.
    :type thread_id: str
    :param message_id: The ID of the message to modify.
    :type message_id: str
    :param modify_message_request: 
    :type modify_message_request: dict | bytes

    :rtype: Union[MessageObject, Tuple[MessageObject, int], Tuple[MessageObject, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        modify_message_request = ModifyMessageRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def modify_run(thread_id, run_id, modify_run_request):  # noqa: E501
    """Modifies a run.

     # noqa: E501

    :param thread_id: The ID of the [thread](/docs/api-reference/threads) that was run.
    :type thread_id: str
    :param run_id: The ID of the run to modify.
    :type run_id: str
    :param modify_run_request: 
    :type modify_run_request: dict | bytes

    :rtype: Union[RunObject, Tuple[RunObject, int], Tuple[RunObject, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        modify_run_request = ModifyRunRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def modify_thread(thread_id, modify_thread_request):  # noqa: E501
    """Modifies a thread.

     # noqa: E501

    :param thread_id: The ID of the thread to modify. Only the &#x60;metadata&#x60; can be modified.
    :type thread_id: str
    :param modify_thread_request: 
    :type modify_thread_request: dict | bytes

    :rtype: Union[ThreadObject, Tuple[ThreadObject, int], Tuple[ThreadObject, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        modify_thread_request = ModifyThreadRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def submit_tool_ouputs_to_run(thread_id, run_id, submit_tool_outputs_run_request):  # noqa: E501
    """When a run has the &#x60;status: \&quot;requires_action\&quot;&#x60; and &#x60;required_action.type&#x60; is &#x60;submit_tool_outputs&#x60;, this endpoint can be used to submit the outputs from the tool calls once they&#39;re all completed. All outputs must be submitted in a single request. 

     # noqa: E501

    :param thread_id: The ID of the [thread](/docs/api-reference/threads) to which this run belongs.
    :type thread_id: str
    :param run_id: The ID of the run that requires the tool output submission.
    :type run_id: str
    :param submit_tool_outputs_run_request: 
    :type submit_tool_outputs_run_request: dict | bytes

    :rtype: Union[RunObject, Tuple[RunObject, int], Tuple[RunObject, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        submit_tool_outputs_run_request = SubmitToolOutputsRunRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'
