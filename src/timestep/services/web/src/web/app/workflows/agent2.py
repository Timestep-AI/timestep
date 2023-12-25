import asyncio
import base64
import os
from logging import Logger

import kubernetes
import sky
import yaml
from prefect import Flow, State, Task, flow, task
from prefect.flows import FlowRun
from prefect.logging import get_run_logger
from prefect.tasks import TaskRun
from prefect_shell import ShellOperation


def load_kubeconfig(overwrite=True):
    kubeconfig_path = os.path.expanduser(sky.clouds.kubernetes.CREDENTIAL_PATH)

    if overwrite or not os.path.exists(kubeconfig_path):
        kubecontext = os.getenv("KUBECONTEXT", "timestep.local")
        kubernetes_service_host = os.getenv("KUBERNETES_SERVICE_HOST")
        kubernetes_service_port = os.getenv("KUBERNETES_SERVICE_PORT")

        ca_certificate_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        namespace_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        service_account_token_path = (
            "/var/run/secrets/kubernetes.io/serviceaccount/token"  # noqa: E501
        )

        with open(namespace_path, "r") as file:
            namespace = file.read()

        # print("before load_incluster_config")
        kubernetes.config.load_incluster_config()
        config = kubernetes.client.Configuration.get_default_copy()

        # kube_config_loader = kubernetes.config.kube_config.KubeConfigLoader(
        #     config_dict=config.to_dict()
        # )

        # print("config", config)
        # kube_config_contexts = kubernetes.config.list_kube_config_contexts()
        # print('kube_config_contexts', kube_config_contexts)

        # kube_config = kubernetes.config.load_kube_config(
        #     client_configuration=config,
        # )
        # print('kube_config', kube_config)

        # api_key = config.api_key
        # print("api_key", api_key)

        # auth = config.auth_settings()
        # print("auth", auth)

        # api_key_prefix = config.api_key_prefix
        # print("api_key_prefix", api_key_prefix)

        # cert_file = config.cert_file
        # print("cert_file", cert_file)

        # key_file = config.key_file
        # print("key_file", key_file)

        # username = config.username
        # print("username", username)

        # password = config.password
        # print("password", password)

        server = config.host
        # print("server", server)
        assert (
            server == f"https://{kubernetes_service_host}:{kubernetes_service_port}"
        ), f"{server} != https://{kubernetes_service_host}:{kubernetes_service_port}"

        ssl_ca_cert = config.ssl_ca_cert
        # print("ssl_ca_cert", ssl_ca_cert)
        assert (
            ssl_ca_cert == ca_certificate_path
        ), f"{ssl_ca_cert} != {ca_certificate_path}"

        # Load CA certificate and encode it in base64
        with open(ssl_ca_cert, "rb") as ssl_ca_cert_file:
            certificate_authority_data = base64.b64encode(
                ssl_ca_cert_file.read()
            ).decode("utf-8")

        # Load service account token
        with open(service_account_token_path, "rb") as token_file:
            service_account_token = token_file.read()

        cluster_name = "timestep.local"
        user_name = "ubuntu"

        # Create kubeconfig dictionary
        kubeconfig = {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [
                {
                    "cluster": {
                        "certificate-authority-data": certificate_authority_data,
                        "server": server,
                    },
                    "name": cluster_name,
                }
            ],
            "contexts": [
                {
                    "context": {
                        "cluster": cluster_name,
                        "namespace": namespace,
                        "user": user_name,
                    },
                    "name": kubecontext,
                }
            ],
            "current-context": kubecontext,
            "preferences": {},
            "users": [
                {
                    "name": user_name,
                    "user": {
                        # "client-certificate-data": client_certificate_data,
                        # "client-key-data": client_key_data,
                        "token": service_account_token
                        # "token-data": service_account_token,
                        # "token": token_file,
                    },
                }
            ],
        }

        # Create ~/.kube directory if it doesn't exist
        kube_dir = os.path.dirname(kubeconfig_path)
        os.makedirs(kube_dir, exist_ok=True)

        # Save the kubeconfig dictionary to ~/.kube/config
        with open(kubeconfig_path, "w") as outfile:
            yaml.dump(kubeconfig, outfile, default_flow_style=False)

        if not os.path.exists(kubeconfig_path):
            raise RuntimeError(f"{kubeconfig_path} file has not been generated.")

        print(f"{kubeconfig_path} file has been generated.")

        # with open(kubeconfig_path, "r") as file:
        # content = file.read()
        # print(f"{kubeconfig_path}:")
        # print(content)


@task
async def load_cloud_credentials():
    load_kubeconfig()


@task
async def sky_check_task():
    get_run_logger()

    async with ShellOperation(
        commands=[
            "sky check",
        ],
        # working_dir=f"data/{today}",
    ) as sky_check_operation:
        # trigger runs the process in the background
        sky_check_operation_process = await sky_check_operation.trigger()

        # then do other things here in the meantime, like download another file
        # logger.info("=== sky_check_operation_process ===")

        # when you're ready, wait for the process to finish
        await sky_check_operation_process.wait_for_completion()

        # if you'd like to get the output lines, you can use the `fetch_result` method
        # output_lines = await sky_check_operation_process.fetch_result()

        # logger.info("output_lines: %s", output_lines)

        # return await sky_check_operation.close()


def sky_launch_task_on_failure_hook(task: Task, task_run: TaskRun, state: State):
    get_run_logger()


@task(
    log_prints=True,
    on_failure=[
        sky_launch_task_on_failure_hook,
    ],
)
async def sky_launch_task():
    get_run_logger()

    async with ShellOperation(
        commands=[
            "sky launch --cluster min --yes examples/minimal.yaml",
        ],
        working_dir="workflows",
    ) as sky_launch_operation:
        sky_launch_operation_process = await sky_launch_operation.trigger()
        await sky_launch_operation_process.wait_for_completion()


def sky_serve_up_task_on_failure_hook(task: Task, task_run: TaskRun, state: State):
    get_run_logger()


@task(
    log_prints=True,
    on_failure=[
        sky_serve_up_task_on_failure_hook,
    ],
)
async def sky_serve_up_task():
    get_run_logger()

    async with ShellOperation(
        commands=[
            "sky serve up --yes examples/serve/ray_serve/ray_serve.yaml",
        ],
        working_dir="workflows",
    ) as sky_serve_up_operation:
        sky_serve_up_operation_process = await sky_serve_up_operation.trigger()
        await sky_serve_up_operation_process.wait_for_completion()


def deploy_agent_flow_on_crashed_hook(flow: Flow, flow_run: FlowRun, state: State):
    get_run_logger()


@flow(
    log_prints=True,
    on_crashed=[
        deploy_agent_flow_on_crashed_hook,
    ],
)
async def deploy_agent_flow(model_ids: list = None):
    logger: Logger = get_run_logger()
    in_cluster = True

    try:
        kubernetes.config.load_incluster_config()

    except kubernetes.config.config_exception.ConfigException as e:
        logger.debug(e)
        in_cluster = False

    if in_cluster:
        await load_cloud_credentials()

    await sky_check_task()
    await sky_launch_task()  # TODO: sky launch -c min minimal.yaml
    # await sky_serve_up_task()


if __name__ == "__main__":
    asyncio.run(
        deploy_agent_flow(
            model_ids=[47],
        )
    )
