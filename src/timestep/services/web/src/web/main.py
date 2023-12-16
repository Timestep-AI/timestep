import logging
import typing
from threading import Thread
from typing import Annotated

import requests

# import kubernetes
# from web.services.users import UserService
# import sky
import strawberry

# import uvicorn  # noqa: F401
# import yaml
# from email_validator import EmailNotValidError, validate_email
from fastapi import Depends, FastAPI, Response
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from llama_index.llms.base import MessageRole

# import base64
# import os
# import typing
from pydantic import BaseModel, Field

# from minio import Minio
# from sky import clouds, skypilot_config
# from sky.adaptors.minio import MINIO_CREDENTIALS_PATH, MINIO_PROFILE_NAME
# from sky.check import check as sky_check
# from sky.check import get_cloud_credential_file_mounts  # noqa: F401
# from sky.cloud_stores import CloudStorage
# from sky.data import Storage, StorageMode, StoreType, data_utils, storage  # noqa: F401, E501
# from sky.serve import service_spec  # noqa: F401
# from sky.serve.core import up as sky_serve_up  # noqa: F401
# from sky.skypilot_config import CONFIG_PATH, _try_load_config
from strawberry.fastapi import GraphQLRouter

# from .api import agent
# from .db.env import envs_by_id

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# logger = logging.getLogger("uvicorn")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# agents_by_id = {
#     "default": {
#         "agent_id": "default",
#     },
# }


# @strawberry.type
# class Agent:
#     agent_id: str


# @strawberry.type
# class Environment:
#     env_id: str
#     agent_ids: typing.List[str]


# @strawberry.type
# class SignUpSuccess:
#     # user_id: int
#     user_id: str


# @strawberry.type
# class SignUpError:
#     message: str


# SignUpResult = Annotated[
#     Union[SignUpSuccess, SignUpError], strawberry.union("SignUpResult")
# ]


# def get_agent(root, agent_id: strawberry.ID) -> Agent:
#     return Agent(**agents_by_id[agent_id])


# def get_agents(root) -> typing.List[Agent]:
#     return [get_agent(root, agent_id) for agent_id in ["default"]]


# def get_env(root, env_id: strawberry.ID) -> Environment:
#     return Environment(**envs_by_id[env_id])


# def get_envs(root) -> typing.List[Environment]:
#     return [get_env(root, env_id) for env_id in envs_by_id.keys()]


# class Context(BaseContext):
#     @cached_property
#     def user(self) -> User | None:
#         if not self.request:
#             return None

#         authorization = self.request.headers.get("Authorization", None)
#         return authorization_service.authorize(authorization)


# Info = _Info[Context, RootValueType]

# async def get_context() -> Context:
#     return Context()

# @strawberry.type
# class Message:
#     id: str
#     text: str
#     timestamp: str
#     thread_id: str
#     avatar_url: str
#     user_id: str

# async def get_message(root, id: strawberry.ID) -> Message:
#     return Message(
#         id=id,
#         text="Hello, world!",
#         thread_id="1",
#     )

# async def get_messages(root, thread_id: strawberry.ID) -> typing.List[Message]:
#     return [
#         Message(
#             id="1",
#             text="Hello, there!",
#             timestamp="2022-01-01T00:00:00Z",
#             thread_id="1",
#             avatar_url="https://www.askmarvin.ai/img/logos/askmarvin_mascot.jpeg",
#             user_id="1",
#         ),
#         Message(
#             id="2",
#             text="How can I help you?",
#             timestamp="2022-01-01T00:00:00Z",
#             thread_id="1",
#             avatar_url="https://www.askmarvin.ai/img/logos/askmarvin_mascot.jpeg",
#             user_id="1",
#         ),
#     ]


@strawberry.type
class MessageThread:
    id: str
    message_ids: typing.List[str]


async def get_thread(root, id: strawberry.ID) -> Thread:
    return MessageThread(
        id=id,
        message_ids=["1", "2", "3"],
    )


async def get_threads(root) -> typing.List[Thread]:
    return [
        MessageThread(
            id="1",
            message_ids=["1", "2", "3"],
        )
    ]


@strawberry.type
class Query:
    # agent: Agent = strawberry.field(resolver=get_agent)
    # agents: typing.List[Agent] = strawberry.field(resolver=get_agents)
    # env: Environment = strawberry.field(resolver=get_env)
    # envs: typing.List[Environment] = strawberry.field(resolver=get_envs)
    # @strawberry.field
    # message: Message = strawberry.field(resolver=get_message)
    # messages: typing.List[Message] = strawberry.field(resolver=get_messages)
    thread: MessageThread = strawberry.field(resolver=get_thread)
    threads: typing.List[MessageThread] = strawberry.field(resolver=get_threads)
    # async def threads(self) -> typing.Dict[str, typing.Any]:
    #     await asyncio.sleep(1)

    #     return {
    #         "list": [{
    #             "id": "1",
    #             "messageIds": ["1", "2", "3"],
    #         }]
    #     }


# @strawberry.type
# class SendMessageInput:
#     text: str


@strawberry.type
class SendMessageOutput:
    id: str


# @strawberry.type
# class ChatData:
#     messages: typing.List[Message]

# @strawberry.type
# class _Message:
#     role: str
#     content: str

# @strawberry.enum
# class MessageRole(MessageRoleEnum):
#     pass

# class MessageRole(str, Enum):
#     ASSISTANT = "assistant"
#     FUNCTION = "function"
#     SYSTEM = "system"
#     TOOL = "tool"
#     USER = "user"


@strawberry.input
class Message:
    content: str
    # role: MessageRole
    role: strawberry.enum(MessageRole)


@strawberry.input
class SendMessageInput:
    # title: str
    # author: str
    # messages: typing.List[Message]
    # things: typing.List[str]
    messages: typing.List[Message]


# @strawberry.type
# class Book:
#     title: str

# @strawberry.input
# class SendMessageInput:
#     messages: typing.List[Message]


@strawberry.type
class Mutation:
    # @strawberry.mutation(extensions=[InputMutationExtension()])
    # @strawberry.mutation()
    # def send_message(self, text: str) -> SendMessageOutput:
    # @strawberry.field
    # def send_message(self, message: SendMessageInput) -> SendMessageOutput:
    #     return SendMessageOutput(id="1")

    @strawberry.field
    async def send_message(self, input: SendMessageInput) -> SendMessageOutput:
        return SendMessageOutput(id="1")

    #     logger = logging.getLogger("uvicorn")
    #     # check preconditions and get last message
    #     if len(input.messages) == 0:
    #         raise HTTPException(
    #             status_code=status.HTTP_400_BAD_REQUEST,
    #             detail="No messages provided",
    #         )

    #     last_message = input.messages.pop()
    #     if last_message.role != MessageRole.USER:
    #         raise HTTPException(
    #             status_code=status.HTTP_400_BAD_REQUEST,
    #             detail="Last message must be from user",
    #         )

    #     # convert messages coming from the request to type ChatMessage
    #     messages = [
    #         ChatMessage(
    #             role=m.role,
    #             content=m.content,
    #         )
    #         for m in input.messages
    #     ]

    #     # agent: OpenAIAgent = get_agent()
    #     # if not agent:
    #     #     raise HTTPException(
    #     #         status_code=status.HTTP_404_NOT_FOUND,
    #     #         detail="Agent not found",
    #     #     )

    #     from transformers import AutoModelForCausalLM, AutoTokenizer
    #     import torch

    #     print('=== info ===')
    #     print('\ttorch.__version__', torch.__version__)
    #     print('\ttorch.cuda.is_available()', torch.cuda.is_available())

    #     # model_id = "EleutherAI/gpt-j-6B"
    #     model_id = "susnato/phi-1_5_dev"
    #     # revision = "float16"  # use float16 weights to fit in 16GB GPUs

    #     # model = AutoModelForCausalLM.from_pretrained(
    #     #     model_id,
    #     #     # revision=revision,
    #     #     torch_dtype=torch.float16,
    #     #     low_cpu_mem_usage=True,
    #     #     device_map="auto",  # automatically makes use of all GPUs available to the Actor  # noqa: E501
    #     # )

    #     print('=== loaded model ===')

    #     # # start a new thread here to query chat engine
    #     # thread = Thread(target=agent.stream_chat, args=(last_message.content, messages))  # noqa: E501
    #     # thread.start()
    #     # logger.info("Querying chat engine")
    #     # # response = agent.stream_chat(last_message.content, messages)

    #     # # logger.info("Querying chat engine done")
    #     # # logger.info("response", response)

    #     # # stream response
    #     # # NOTE: changed to sync due to issues with blocking the event loop
    #     # # see https://stackoverflow.com/a/75760884
    #     # def event_generator():
    #     #     queue = agent.callback_manager.handlers[0].queue

    #     #     # stream response
    #     #     while True:
    #     #         next_item = queue.get(True, 60.0)  # set a generous timeout of 60 seconds  # noqa: E501
    #     #         # check type of next_item, if string or not
    #     #         if isinstance(next_item, EventObject):
    #     #             logger.info('got EventObject')
    #     #             sse = convert_sse(dict(next_item))
    #     #             logger.info("sse", sse)
    #     #             yield sse
    #     #         elif isinstance(next_item, StreamingAgentChatResponse):
    #     #             logger.info('got StreamingAgentChatResponse')
    #     #             response = cast(StreamingAgentChatResponse, next_item)
    #     #             for token in response.response_gen:
    #     #                 sse = convert_sse(token)
    #     #                 logger.info("sse", sse)
    #     #                 yield sse
    #     #             break

    #     # return StreamingResponse(event_generator(), media_type="text/event-stream")

    #     return SendMessageOutput(id="1")

    # @strawberry.field
    # async def sign_up(self, email: str, password: str) -> SignUpResult:
    #     # Your domain-specific authentication logic would go here
    #     # db_url = 'sqlite:///example.db'
    #     # db_url = os.getenv("POSTGRES_CONNECTION_STRING")
    #     # if db_url is None:
    #     # return SignUpError(message="POSTGRES_CONNECTION_STRING not set")

    #     # user_service = UserService(db_url)
    #     # user_service: UserService = self.request.app.state.user_service
    #     # data = await self.request.json()
    #     # email = data.get("email")
    #     # password = data.get("password")

    #     # user_id = await user_service.create_user(email, password)
    #     # return {"message": "User created successfully!"}

    #     # remote_schema = strawberry.Schema(
    #     #     query=Query,
    #     #     mutation=Mutation,
    #     # )

    #     transport = AIOHTTPTransport(url="https://countries.trevorblades.com/graphql")

    #     # Using `async with` on the client will start a connection on the transport
    #     # and provide a `session` variable to execute queries on this connection
    #     async with Client(
    #         transport=transport,
    #         fetch_schema_from_transport=True,
    #     ) as session:
    #         # Execute single query
    #         query = gql(
    #             """
    #             query getContinents {
    #             continents {
    #                 code
    #                 name
    #             }
    #             }
    #         """
    #         )

    #         result = await session.execute(query)
    #         print(result)

    #     # transport = AIOHTTPTransport(url="http://hasura-graphql-engine:8080/v1/graphql")

    #     # async with Client(
    #     #     transport=transport,
    #     #     fetch_schema_from_transport=True,
    #     # ) as session:
    #     #     query = gql(
    #     #         """
    #     #         fragment userFields on users {
    #     #         id
    #     #         createdAt
    #     #         disabled
    #     #         displayName
    #     #         avatarUrl
    #     #         email
    #     #         passwordHash
    #     #         emailVerified
    #     #         phoneNumber
    #     #         phoneNumberVerified
    #     #         defaultRole
    #     #         isAnonymous
    #     #         ticket
    #     #         otpHash
    #     #         totpSecret
    #     #         activeMfaType
    #     #         newEmail
    #     #         locale
    #     #         metadata
    #     #         roles {
    #     #             role
    #     #         }
    #     #         }

    #     #         mutation insertUser($user: users_insert_input!) {
    #     #         insertUser(object: $user) {
    #     #             ...userFields
    #     #         }

    #     #         {
    #     #             "user": ...
    #     #         }
    #     #         }
    #     #     """
    #     #     )

    #     #     result = await session.execute(query)
    #     #     print(result)

    #     user_id = result

    #     if user_id is None:
    #         return SignUpError(message="Something went wrong")

    #     return SignUpSuccess(user_id=user_id)


schema = strawberry.Schema(
    # context_getter=get_context,
    query=Query,
    mutation=Mutation,
)

graphql_app = GraphQLRouter(
    schema=schema,
    graphiql=True,
    allow_queries_via_get=False,
)

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    logger.debug("=== startup_event (BEGIN) ===")
    # app.state.user_service = await init()
    # app.state.storage_service = await init_storage_service()
    # app.state.user_index_service = await init_user_index_service()

    # @flow(log_prints=True)
    # def get_repo_info(repo_name: str = "PrefectHQ/prefect"):
    #     url = f"https://api.github.com/repos/{repo_name}"
    #     response = httpx.get(url)
    #     response.raise_for_status()
    #     repo = response.json()
    #     print(f"{repo_name} repository statistics 🤓:")
    #     print(f"Stars 🌠 : {repo['stargazers_count']}")
    #     print(f"Forks 🍴 : {repo['forks_count']}")

    # # await get_repo_info.deploy(
    # #     name="my-first-deployment",
    # #     work_pool_name="default-worker-pool",
    # #     # image="my-first-deployment-image:tutorial",
    # #     image="registry.gitlab.com/timestep-ai/timestep/web:latest",
    # #     push=False
    # # )

    # await deploy(
    #     get_repo_info.to_deployment(name="my-first-deployment"),
    #     build=False,
    #     image="registry.gitlab.com/timestep-ai/timestep/web:latest",
    #     push=False,
    #     work_pool_name="default-worker-pool",
    # )

    # print("=== startup complete ===")
    logger.debug("=== startup_event (END) ===")


# async def get_ready_flow_on_cancellation(flow, flow_run, state):
#     print("=== get_ready_flow_on_cancellation ===")

# async def get_ready_flow_on_crashed(flow, flow_run, exc):
#     print("=== get_ready_flow_on_crashed ===")

# async def get_ready_flow_on_failure(flow, flow_run, exc):
#     print("=== get_ready_flow_on_failure ===")


# @flow(
#     log_prints=True,
#     on_cancellation=[get_ready_flow_on_cancellation],
#     on_crashed=[get_ready_flow_on_crashed],
#     on_failure=[get_ready_flow_on_failure],
#     retries=3,
# )
def get_ready_flow():
    return {
        "ready": "okay",
    }


@app.get("/ready")
async def get_ready():
    return get_ready_flow()


security = HTTPBearer()


@app.post("/api/accounts")
async def create_account(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
):
    logger.debug("=== create_account ===")

    # return {"scheme": credentials.scheme, "credentials": credentials.credentials}

    response = requests.get(
        "http://nhost-hasura-auth:4000/mfa/totp/generate",
        headers={
            "Authorization": f"Bearer {credentials.credentials}",
        },
    )

    return response.json()


class Mfa(BaseModel):
    ticket: str = Field("")


class User(BaseModel):
    activate_mfa_type: str = Field("", alias="activateMfaType")
    mfa: Mfa = Field(None, alias="mfa")
    totp_code: str = Field("", alias="totpCode")


@app.put("/api/users/{user_id}")
async def update_user(
    user_id: str,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user: User,
):
    logger.debug("=== update_user ===")
    logger.debug(f"user: {user}")

    if user.totp_code:
        if user.activate_mfa_type == "totp":
            response = requests.post(
                "http://nhost-hasura-auth:4000/user/mfa",
                # auth=Auth,
                json={
                    "code": user.totp_code,
                    "activeMfaType": user.activate_mfa_type,
                },
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                },
            )

        else:
            response = requests.post(
                "http://nhost-hasura-auth:4000/signin/mfa/totp",
                json={
                    # "code": user.totp_code,
                    "otp": user.totp_code,
                    "ticket": user.mfa.ticket,
                },
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                },
            )

    # return response.json()
    return Response(
        content=response.content,
        status_code=response.status_code,
    )


@app.delete("/api/accounts/{account_id}")
async def delete_account(account_id: str):
    logger.debug("=== delete_account ===")
    logger.debug(f"account_id: {account_id}")

    return {
        "account_id": account_id,
    }


# @flow(log_prints=True, name='test-flow')
# def get_repo_info(repo_name: str = "PrefectHQ/prefect"):
#     url = f"https://api.github.com/repos/{repo_name}"
#     response = httpx.get(url)
#     response.raise_for_status()
#     repo = response.json()
#     print(f"{repo_name} repository statistics 🤓:")
#     print(f"Stars 🌠 : {repo['stargazers_count']}")
#     print(f"Forks 🍴 : {repo['forks_count']}")

# @flow(log_prints=True)
# def buy():
#     print("Buying securities")

# @app.get("/test")
# def test(background_tasks: BackgroundTasks):
#     # from transformers import AutoModelForCausalLM, AutoTokenizer
#     # import torch

#     # print('=== test ===', flush=True)
#     logger.debug('=== test ===')
#     logger.debug(f'torch.__version__ {torch.__version__}')
#     logger.debug(f'torch.cuda.is_available() {torch.cuda.is_available()}')

#     # id = deploy_flow()
#     id = background_tasks.add_task(deploy_flow)

#     # model_id = "EleutherAI/gpt-j-6B"
#     # model_id = "susnato/phi-1_5_dev"
#     # revision = "float16"  # use float16 weights to fit in 16GB GPUs

#     # model = AutoModelForCausalLM.from_pretrained(
#     #     model_id,
#     #     # revision=revision,
#     #     torch_dtype=torch.float16,
#     #     low_cpu_mem_usage=True,
#     #     device_map="auto",  # automatically makes use of all GPUs available to the Actor  # noqa: E501
#     # )

#     # get_repo_info.deploy(
#     #     "test-flow-deployment",
#     #     build=False,
#     #     # image="my-first-deployment-image:tutorial",
#     #     # image="registry.gitlab.com/timestep-ai/timestep/web:latest",
#     #     image="prefecthq/prefect:2.14.6-python3.11-kubernetes",
#     #     push=False,
#     #     work_pool_name="default-worker-pool",
#     # )

#     # buy.deploy(
#     #     name="my-code-baked-into-an-image-deployment",
#     #     work_pool_name="my-docker-pool",
#     #     image="my_registry/my_image:my_image_tag"
#     # )

#     # await deploy(
#     #     get_repo_info.to_deployment(name="my-first-deployment"),
#     #     build=False,
#     #     # image="registry.gitlab.com/timestep-ai/timestep/web:latest",
#     #     image="prefecthq/prefect:2.14.6-python3.11-kubernetes",
#     #     push=False,
#     #     work_pool_name="default-worker-pool",
#     # )

#     return {
#         "test": id,
#     }

# @app.post("/query")
# async def query_index(query: str):
#     return {
#         "query": query,
#     }

# @app.get("/query")
# async def query_index(query: str):
#     return await app.state.user_index_service.query_index(query)

# class CreateUserRequestBody(BaseModel):
#     email: str
#     password: str


# @app.post("/rest/users")
# async def create_user(
#     create_user_request_body: CreateUserRequestBody,
#     request: Request,
# ):
#     user_service: UserService = request.app.state.user_service
#     # data = await request.json()
#     # email = data.get("email")
#     # password = data.get("password")

#     email = create_user_request_body.email
#     password = create_user_request_body.password

#     try:
#         email_info = validate_email(email, check_deliverability=False)
#         email = email_info.normalized

#     except EmailNotValidError as e:
#         return {"message": str(e)}

#     await user_service.create_user(email, password)
#     return {"message": "User created successfully!"}


# class MinioCloudStorage(CloudStorage):
#     """MinIO Cloud Storage."""

#     def __init__(self, endpoint, access_key, secret_key):
#         self.minio_client = Minio(endpoint, access_key, secret_key, secure=False)

#     def is_directory(self, url: str) -> bool:
#         """Returns whether MinIO 'url' is a directory."""
#         bucket_name, path = data_utils.split_s3_path(url)

#         try:
#             # Attempt to retrieve the object. If it exists, it's a file; otherwise, it's a directory.  # noqa: E501
#             self.minio_client.stat_object(bucket_name, path)
#             return False

#         # except NoSuchKey:
#         except Exception as e:
#             print(e)
#             return True

#     def make_sync_dir_command(self, source: str, destination: str) -> str:
#         """Downloads from MinIO."""
#         sync_command = f"mc mirror {source} {destination}"

#         return sync_command

#     def make_sync_file_command(self, source: str, destination: str) -> str:
#         """Downloads a file from MinIO."""
#         cp_command = f"mc cp {source} {destination}"

#         return cp_command


# @app.get("/sky")
# def get_sky():
#     cloud_info = {}

#     for cloud_name, cloud in sky.clouds.CLOUD_REGISTRY.items():
#         cloud_info[cloud_name] = {
#             "enabled": False,
#             "name": str(cloud),
#         }

#     data = {
#         "cloud_info": cloud_info,
#         "commit": sky.__commit__,
#         "version": sky.__version__,
#         "root_dir": sky.__root_dir__,
#     }

#     print("=== load_cloud_credentials? ===")
#     load_cloud_credentials()

#     try:
#         sky_check(
#             quiet=False,
#             verbose=True,
#         )

#         enabled_clouds = sky.global_user_state.get_enabled_clouds()

#         for cloud in enabled_clouds:
#             data["cloud_info"][str(cloud).lower()]["enabled"] = True

#     except SystemExit:
#         data["error"] = {
#             "type": "SystemExit",
#         }

#     # store = storage.Storage(

#     # TEST_BUCKET_NAME = 'skypilot-workdir-ubuntu-b0670fb3'
#     # LOCAL_SOURCE_PATH = '/home/ubuntu/app/src/web/examples/serve/ray_serve'
#     # storage_1 = storage.Storage(name=TEST_BUCKET_NAME, source=LOCAL_SOURCE_PATH)
#     # # storage_1.add_store(StoreType.S3)  # Transfers data from local to S3
#     # storage_1.add_store(StoreType.MINIO)

#     # storages = sky.core.storage_ls()
#     # print('storages', storages)

#     task = (
#         sky.Task(
#             run='echo "Hello, how are you?',
#             # run='serve run serve:app --host 0.0.0.0',
#             setup='echo "Running setup."',
#             # setup='pip install "ray[serve]"',
#             workdir=".",
#             # workdir=f'{os.getcwd()}/src/web/examples/serve/ray_serve',
#         )
#         .set_file_mounts(
#             {
#                 "/dataset-demo": "minio://sky-demo-dataset",
#             }
#         )
#         .set_resources(
#             sky.Resources(
#                 cloud=clouds.Kubernetes(),
#                 cpus="1",
#                 # cpus='2+',
#                 memory="2",
#                 # ports='8000',
#             )
#             # ).set_service(
#             #     service_spec.SkyServiceSpec(
#             #         initial_delay_seconds=5,
#             #         min_replicas=1,
#             #         readiness_path='/',
#             #     )
#         )
#     )
#     # ).set_storage_mounts( #  Workdir '/home/ubuntu/app/src/web/examples/serve/ray_serve' will be synced to cloud storage 'skypilot-workdir-ubuntu-b0670fb3'.  # noqa: E501
#     #     {
#     #         f"{mount_path}": sky.Storage(
#     #             name="skypilot-workdir-ubuntu-b0670fb3",
#     #             source="/home/ubuntu/app/src/web/examples/serve/ray_serve",
#     #         )
#     #     }

#     # # sky serve up examples/serve/ray_serve/ray_serve.yaml
#     # sky_serve_up(
#     #     service_name=None,
#     #     task=task,
#     # )

#     job_id, handle = sky.launch(
#         cluster_name="sky-5cf0-ubuntu",
#         task=task,
#     )

#     return {
#         "job_id": job_id,
#         "sky": data,
#     }


# def load_cloud_credentials(overwrite=True):
#     load_kubeconfig(overwrite)
#     load_minio_credentials(overwrite)


# def load_kubeconfig(overwrite=True):
#     kubeconfig_path = os.path.expanduser(sky.clouds.kubernetes.CREDENTIAL_PATH)

#     if overwrite or not os.path.exists(kubeconfig_path):
#         kubecontext = os.getenv("KUBECONTEXT", "timestep.local")
#         kubernetes_service_host = os.getenv("KUBERNETES_SERVICE_HOST")
#         kubernetes_service_port = os.getenv("KUBERNETES_SERVICE_PORT")

#         ca_certificate_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
#         namespace_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
#         service_account_token_path = (
#             "/var/run/secrets/kubernetes.io/serviceaccount/token"  # noqa: E501
#         )

#         with open(namespace_path, "r") as file:
#             namespace = file.read()

#         print("before load_incluster_config")
#         kubernetes.config.load_incluster_config()
#         config = kubernetes.client.Configuration.get_default_copy()

#         # kube_config_loader = kubernetes.config.kube_config.KubeConfigLoader(
#         #     config_dict=config.to_dict()
#         # )

#         print("config", config)
#         # kube_config_contexts = kubernetes.config.list_kube_config_contexts()
#         # print('kube_config_contexts', kube_config_contexts)

#         # kube_config = kubernetes.config.load_kube_config(
#         #     client_configuration=config,
#         # )
#         # print('kube_config', kube_config)

#         api_key = config.api_key
#         print("api_key", api_key)

#         auth = config.auth_settings()
#         print("auth", auth)

#         api_key_prefix = config.api_key_prefix
#         print("api_key_prefix", api_key_prefix)

#         cert_file = config.cert_file
#         print("cert_file", cert_file)

#         key_file = config.key_file
#         print("key_file", key_file)

#         username = config.username
#         print("username", username)

#         password = config.password
#         print("password", password)

#         server = config.host
#         print("server", server)
#         assert (
#             server == f"https://{kubernetes_service_host}:{kubernetes_service_port}"
#         ), f"{server} != https://{kubernetes_service_host}:{kubernetes_service_port}"

#         ssl_ca_cert = config.ssl_ca_cert
#         print("ssl_ca_cert", ssl_ca_cert)
#         assert (
#             ssl_ca_cert == ca_certificate_path
#         ), f"{ssl_ca_cert} != {ca_certificate_path}"

#         # Load CA certificate and encode it in base64
#         with open(ssl_ca_cert, "rb") as ssl_ca_cert_file:
#             certificate_authority_data = base64.b64encode(
#                 ssl_ca_cert_file.read()
#             ).decode("utf-8")

#         # Load service account token
#         with open(service_account_token_path, "rb") as token_file:
#             service_account_token = token_file.read()

#         cluster_name = "timestep.local"
#         user_name = "ubuntu"

#         # Create kubeconfig dictionary
#         kubeconfig = {
#             "apiVersion": "v1",
#             "kind": "Config",
#             "clusters": [
#                 {
#                     "cluster": {
#                         "certificate-authority-data": certificate_authority_data,
#                         "server": server,
#                     },
#                     "name": cluster_name,
#                 }
#             ],
#             "contexts": [
#                 {
#                     "context": {
#                         "cluster": cluster_name,
#                         "namespace": namespace,
#                         "user": user_name,
#                     },
#                     "name": kubecontext,
#                 }
#             ],
#             "current-context": kubecontext,
#             "preferences": {},
#             "users": [
#                 {
#                     "name": user_name,
#                     "user": {
#                         # "client-certificate-data": client_certificate_data,
#                         # "client-key-data": client_key_data,
#                         "token": service_account_token
#                         # "token-data": service_account_token,
#                         # "token": token_file,
#                     },
#                 }
#             ],
#         }

#         # Create ~/.kube directory if it doesn't exist
#         kube_dir = os.path.dirname(kubeconfig_path)
#         os.makedirs(kube_dir, exist_ok=True)

#         # Save the kubeconfig dictionary to ~/.kube/config
#         with open(kubeconfig_path, "w") as outfile:
#             yaml.dump(kubeconfig, outfile, default_flow_style=False)

#         if not os.path.exists(kubeconfig_path):
#             raise RuntimeError(f"{kubeconfig_path} file has not been generated.")

#         print(f"{kubeconfig_path} file has been generated.")

#         with open(kubeconfig_path, "r") as file:
#             content = file.read()
#             print(f"{kubeconfig_path}:")
#             print(content)


# def load_minio_credentials(overwrite=True):
#     minio_credentials_path = os.path.expanduser(MINIO_CREDENTIALS_PATH)
#     minio_credentials = f"""[{MINIO_PROFILE_NAME}]
# aws_access_key_id={os.getenv("MINIO_ROOT_USER")}
# aws_secret_access_key={os.getenv("MINIO_ROOT_PASSWORD")}
# """

#     if overwrite or not os.path.exists(minio_credentials_path):
#         minio_credentials_dir = os.path.dirname(minio_credentials_path)
#         os.makedirs(minio_credentials_dir, exist_ok=True)

#         with open(minio_credentials_path, "w") as outfile:
#             outfile.write(minio_credentials)
#         if not os.path.exists(minio_credentials_path):
#             raise RuntimeError(f"{minio_credentials_path} file has not been generated.")  # noqa: E501

#         with open(minio_credentials_path, "r") as file:
#             content = file.read()
#             print(f"{minio_credentials_path}:")
#             print(content)

#     aws_credentials_path = os.path.expanduser("~/.aws/credentials")
#     aws_credentials = f"""[default]
# aws_access_key_id={os.getenv("AWS_ACCESS_KEY_ID")}
# aws_secret_access_key={os.getenv("AWS_SECRET_ACCESS_KEY")}

# {minio_credentials}
# """

#     if overwrite or not os.path.exists(aws_credentials_path):
#         aws_credentials_dir = os.path.dirname(aws_credentials_path)
#         os.makedirs(aws_credentials_dir, exist_ok=True)

#         with open(aws_credentials_path, "w") as outfile:
#             outfile.write(aws_credentials)
#         if not os.path.exists(aws_credentials_path):
#             raise RuntimeError(f"{aws_credentials_path} file has not been generated.")

#         with open(aws_credentials_path, "r") as file:
#             content = file.read()
#             print(f"{aws_credentials_path}:")
#             print(content)

#     config_path = os.path.expanduser(CONFIG_PATH)
#     config = f"""{MINIO_PROFILE_NAME}:
#     endpoint: "http://minio.default.svc.cluster.local:9000"
# """

#     if overwrite or not os.path.exists(config_path):
#         config_dir = os.path.dirname(config_path)
#         os.makedirs(config_dir, exist_ok=True)

#         with open(config_path, "w") as outfile:
#             outfile.write(config)
#         if not os.path.exists(config_path):
#             raise RuntimeError(f"{config_path} file has not been generated.")

#         with open(config_path, "r") as file:
#             content = file.read()
#             print(f"{config_path}:")
#             print(content)

#     _try_load_config()

#     if not skypilot_config.get_nested(("minio", "endpoint"), None):
#         raise Exception(f"minio endpoint is not set in {config_path}")


# app.include_router(chat_router, prefix="/api/chat")
# app.include_router(graphql_app, prefix="/graphql")

# for env_id in envs_by_id.keys():
#     env = get_env(None, env_id)

#     for agent_id in env.agent_ids:
#         app.include_router(
#             agent.router, prefix=f"/envs/{env.env_id}/agents/{agent_id}"
#         )  # noqa: E501

# # if __name__ == "__main__":
# #     # CMD ["poetry", "run", "uvicorn", "src.web.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "5000"]  # noqa: E501
# #     # uvicorn.run(app, host="0.0.0.0", port=5000, proxy_headers=True, reload=True)
# #     uvicorn.run(app, host="0.0.0.0", port=5000, proxy_headers=True)
