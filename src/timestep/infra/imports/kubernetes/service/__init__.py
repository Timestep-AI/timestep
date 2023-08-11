'''
# `kubernetes_service`

Refer to the Terraform Registory for docs: [`kubernetes_service`](https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service).
'''
import abc
import builtins
import datetime
import enum
import typing

import jsii
import publication
import typing_extensions

from typeguard import check_type

from .._jsii import *

import cdktf as _cdktf_9a9027ec
import constructs as _constructs_77d1e7e8


class Service(
    _cdktf_9a9027ec.TerraformResource,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.Service",
):
    '''Represents a {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service kubernetes_service}.'''

    def __init__(
        self,
        scope: _constructs_77d1e7e8.Construct,
        id_: builtins.str,
        *,
        metadata: typing.Union["ServiceMetadata", typing.Dict[builtins.str, typing.Any]],
        spec: typing.Union["ServiceSpec", typing.Dict[builtins.str, typing.Any]],
        id: typing.Optional[builtins.str] = None,
        timeouts: typing.Optional[typing.Union["ServiceTimeouts", typing.Dict[builtins.str, typing.Any]]] = None,
        wait_for_load_balancer: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
        connection: typing.Optional[typing.Union[typing.Union[_cdktf_9a9027ec.SSHProvisionerConnection, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.WinrmProvisionerConnection, typing.Dict[builtins.str, typing.Any]]]] = None,
        count: typing.Optional[typing.Union[jsii.Number, _cdktf_9a9027ec.TerraformCount]] = None,
        depends_on: typing.Optional[typing.Sequence[_cdktf_9a9027ec.ITerraformDependable]] = None,
        for_each: typing.Optional[_cdktf_9a9027ec.ITerraformIterator] = None,
        lifecycle: typing.Optional[typing.Union[_cdktf_9a9027ec.TerraformResourceLifecycle, typing.Dict[builtins.str, typing.Any]]] = None,
        provider: typing.Optional[_cdktf_9a9027ec.TerraformProvider] = None,
        provisioners: typing.Optional[typing.Sequence[typing.Union[typing.Union[_cdktf_9a9027ec.FileProvisioner, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.LocalExecProvisioner, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.RemoteExecProvisioner, typing.Dict[builtins.str, typing.Any]]]]] = None,
    ) -> None:
        '''Create a new {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service kubernetes_service} Resource.

        :param scope: The scope in which to define this construct.
        :param id_: The scoped construct ID. Must be unique amongst siblings in the same scope
        :param metadata: metadata block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#metadata Service#metadata}
        :param spec: spec block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#spec Service#spec}
        :param id: Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#id Service#id}. Please be aware that the id field is automatically added to all resources in Terraform providers using a Terraform provider SDK version below 2. If you experience problems setting this value it might not be settable. Please take a look at the provider documentation to ensure it should be settable.
        :param timeouts: timeouts block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#timeouts Service#timeouts}
        :param wait_for_load_balancer: Terraform will wait for the load balancer to have at least 1 endpoint before considering the resource created. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#wait_for_load_balancer Service#wait_for_load_balancer}
        :param connection: 
        :param count: 
        :param depends_on: 
        :param for_each: 
        :param lifecycle: 
        :param provider: 
        :param provisioners: 
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__9379740d5a41c67727672ceaa889c18038f8f9e9ea25dc72641d1e42e916ce26)
            check_type(argname="argument scope", value=scope, expected_type=type_hints["scope"])
            check_type(argname="argument id_", value=id_, expected_type=type_hints["id_"])
        config = ServiceConfig(
            metadata=metadata,
            spec=spec,
            id=id,
            timeouts=timeouts,
            wait_for_load_balancer=wait_for_load_balancer,
            connection=connection,
            count=count,
            depends_on=depends_on,
            for_each=for_each,
            lifecycle=lifecycle,
            provider=provider,
            provisioners=provisioners,
        )

        jsii.create(self.__class__, self, [scope, id_, config])

    @jsii.member(jsii_name="putMetadata")
    def put_metadata(
        self,
        *,
        annotations: typing.Optional[typing.Mapping[builtins.str, builtins.str]] = None,
        generate_name: typing.Optional[builtins.str] = None,
        labels: typing.Optional[typing.Mapping[builtins.str, builtins.str]] = None,
        name: typing.Optional[builtins.str] = None,
        namespace: typing.Optional[builtins.str] = None,
    ) -> None:
        '''
        :param annotations: An unstructured key value map stored with the service that may be used to store arbitrary metadata. More info: http://kubernetes.io/docs/user-guide/annotations Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#annotations Service#annotations}
        :param generate_name: Prefix, used by the server, to generate a unique name ONLY IF the ``name`` field has not been provided. This value will also be combined with a unique suffix. Read more: https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#idempotency Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#generate_name Service#generate_name}
        :param labels: Map of string keys and values that can be used to organize and categorize (scope and select) the service. May match selectors of replication controllers and services. More info: http://kubernetes.io/docs/user-guide/labels Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#labels Service#labels}
        :param name: Name of the service, must be unique. Cannot be updated. More info: http://kubernetes.io/docs/user-guide/identifiers#names. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#name Service#name}
        :param namespace: Namespace defines the space within which name of the service must be unique. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#namespace Service#namespace}
        '''
        value = ServiceMetadata(
            annotations=annotations,
            generate_name=generate_name,
            labels=labels,
            name=name,
            namespace=namespace,
        )

        return typing.cast(None, jsii.invoke(self, "putMetadata", [value]))

    @jsii.member(jsii_name="putSpec")
    def put_spec(
        self,
        *,
        allocate_load_balancer_node_ports: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
        cluster_ip: typing.Optional[builtins.str] = None,
        cluster_ips: typing.Optional[typing.Sequence[builtins.str]] = None,
        external_ips: typing.Optional[typing.Sequence[builtins.str]] = None,
        external_name: typing.Optional[builtins.str] = None,
        external_traffic_policy: typing.Optional[builtins.str] = None,
        health_check_node_port: typing.Optional[jsii.Number] = None,
        internal_traffic_policy: typing.Optional[builtins.str] = None,
        ip_families: typing.Optional[typing.Sequence[builtins.str]] = None,
        ip_family_policy: typing.Optional[builtins.str] = None,
        load_balancer_class: typing.Optional[builtins.str] = None,
        load_balancer_ip: typing.Optional[builtins.str] = None,
        load_balancer_source_ranges: typing.Optional[typing.Sequence[builtins.str]] = None,
        port: typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.Sequence[typing.Union["ServiceSpecPort", typing.Dict[builtins.str, typing.Any]]]]] = None,
        publish_not_ready_addresses: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
        selector: typing.Optional[typing.Mapping[builtins.str, builtins.str]] = None,
        session_affinity: typing.Optional[builtins.str] = None,
        session_affinity_config: typing.Optional[typing.Union["ServiceSpecSessionAffinityConfig", typing.Dict[builtins.str, typing.Any]]] = None,
        type: typing.Optional[builtins.str] = None,
    ) -> None:
        '''
        :param allocate_load_balancer_node_ports: Defines if ``NodePorts`` will be automatically allocated for services with type ``LoadBalancer``. It may be set to ``false`` if the cluster load-balancer does not rely on ``NodePorts``. If the caller requests specific ``NodePorts`` (by specifying a value), those requests will be respected, regardless of this field. This field may only be set for services with type ``LoadBalancer``. Default is ``true``. More info: https://kubernetes.io/docs/concepts/services-networking/service/#load-balancer-nodeport-allocation Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#allocate_load_balancer_node_ports Service#allocate_load_balancer_node_ports}
        :param cluster_ip: The IP address of the service. It is usually assigned randomly by the master. If an address is specified manually and is not in use by others, it will be allocated to the service; otherwise, creation of the service will fail. ``None`` can be specified for headless services when proxying is not required. Ignored if type is ``ExternalName``. More info: http://kubernetes.io/docs/user-guide/services#virtual-ips-and-service-proxies Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#cluster_ip Service#cluster_ip}
        :param cluster_ips: List of IP addresses assigned to this service, and are usually assigned randomly. If an address is specified manually and is not in use by others, it will be allocated to the service; otherwise creation of the service will fail. If this field is not specified, it will be initialized from the ``clusterIP`` field. If this field is specified, clients must ensure that ``clusterIPs[0]`` and ``clusterIP`` have the same value. More info: http://kubernetes.io/docs/user-guide/services#virtual-ips-and-service-proxies Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#cluster_ips Service#cluster_ips}
        :param external_ips: A list of IP addresses for which nodes in the cluster will also accept traffic for this service. These IPs are not managed by Kubernetes. The user is responsible for ensuring that traffic arrives at a node with this IP. A common example is external load-balancers that are not part of the Kubernetes system. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#external_ips Service#external_ips}
        :param external_name: The external reference that kubedns or equivalent will return as a CNAME record for this service. No proxying will be involved. Must be a valid DNS name and requires ``type`` to be ``ExternalName``. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#external_name Service#external_name}
        :param external_traffic_policy: Denotes if this Service desires to route external traffic to node-local or cluster-wide endpoints. ``Local`` preserves the client source IP and avoids a second hop for LoadBalancer and Nodeport type services, but risks potentially imbalanced traffic spreading. ``Cluster`` obscures the client source IP and may cause a second hop to another node, but should have good overall load-spreading. More info: https://kubernetes.io/docs/tutorials/services/source-ip/ Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#external_traffic_policy Service#external_traffic_policy}
        :param health_check_node_port: Specifies the Healthcheck NodePort for the service. Only effects when type is set to ``LoadBalancer`` and external_traffic_policy is set to ``Local``. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#health_check_node_port Service#health_check_node_port}
        :param internal_traffic_policy: Specifies if the cluster internal traffic should be routed to all endpoints or node-local endpoints only. ``Cluster`` routes internal traffic to a Service to all endpoints. ``Local`` routes traffic to node-local endpoints only, traffic is dropped if no node-local endpoints are ready. The default value is ``Cluster``. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#internal_traffic_policy Service#internal_traffic_policy}
        :param ip_families: IPFamilies is a list of IP families (e.g. IPv4, IPv6) assigned to this service. This field is usually assigned automatically based on cluster configuration and the ipFamilyPolicy field. If this field is specified manually, the requested family is available in the cluster, and ipFamilyPolicy allows it, it will be used; otherwise creation of the service will fail. This field is conditionally mutable: it allows for adding or removing a secondary IP family, but it does not allow changing the primary IP family of the Service. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#ip_families Service#ip_families}
        :param ip_family_policy: IPFamilyPolicy represents the dual-stack-ness requested or required by this Service. If there is no value provided, then this field will be set to SingleStack. Services can be 'SingleStack' (a single IP family), 'PreferDualStack' (two IP families on dual-stack configured clusters or a single IP family on single-stack clusters), or 'RequireDualStack' (two IP families on dual-stack configured clusters, otherwise fail). The ipFamilies and clusterIPs fields depend on the value of this field. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#ip_family_policy Service#ip_family_policy}
        :param load_balancer_class: The class of the load balancer implementation this Service belongs to. If specified, the value of this field must be a label-style identifier, with an optional prefix. This field can only be set when the Service type is ``LoadBalancer``. If not set, the default load balancer implementation is used. This field can only be set when creating or updating a Service to type ``LoadBalancer``. More info: https://kubernetes.io/docs/concepts/services-networking/service/#load-balancer-class Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#load_balancer_class Service#load_balancer_class}
        :param load_balancer_ip: Only applies to ``type = LoadBalancer``. LoadBalancer will get created with the IP specified in this field. This feature depends on whether the underlying cloud-provider supports specifying this field when a load balancer is created. This field will be ignored if the cloud-provider does not support the feature. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#load_balancer_ip Service#load_balancer_ip}
        :param load_balancer_source_ranges: If specified and supported by the platform, this will restrict traffic through the cloud-provider load-balancer will be restricted to the specified client IPs. This field will be ignored if the cloud-provider does not support the feature. More info: http://kubernetes.io/docs/user-guide/services-firewalls Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#load_balancer_source_ranges Service#load_balancer_source_ranges}
        :param port: port block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#port Service#port}
        :param publish_not_ready_addresses: When set to true, indicates that DNS implementations must publish the ``notReadyAddresses`` of subsets for the Endpoints associated with the Service. The default value is ``false``. The primary use case for setting this field is to use a StatefulSet's Headless Service to propagate ``SRV`` records for its Pods without respect to their readiness for purpose of peer discovery. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#publish_not_ready_addresses Service#publish_not_ready_addresses}
        :param selector: Route service traffic to pods with label keys and values matching this selector. Only applies to types ``ClusterIP``, ``NodePort``, and ``LoadBalancer``. More info: http://kubernetes.io/docs/user-guide/services#overview Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#selector Service#selector}
        :param session_affinity: Used to maintain session affinity. Supports ``ClientIP`` and ``None``. Defaults to ``None``. More info: http://kubernetes.io/docs/user-guide/services#virtual-ips-and-service-proxies. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#session_affinity Service#session_affinity}
        :param session_affinity_config: session_affinity_config block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#session_affinity_config Service#session_affinity_config}
        :param type: Determines how the service is exposed. Defaults to ``ClusterIP``. Valid options are ``ExternalName``, ``ClusterIP``, ``NodePort``, and ``LoadBalancer``. ``ExternalName`` maps to the specified ``external_name``. More info: http://kubernetes.io/docs/user-guide/services#overview Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#type Service#type}
        '''
        value = ServiceSpec(
            allocate_load_balancer_node_ports=allocate_load_balancer_node_ports,
            cluster_ip=cluster_ip,
            cluster_ips=cluster_ips,
            external_ips=external_ips,
            external_name=external_name,
            external_traffic_policy=external_traffic_policy,
            health_check_node_port=health_check_node_port,
            internal_traffic_policy=internal_traffic_policy,
            ip_families=ip_families,
            ip_family_policy=ip_family_policy,
            load_balancer_class=load_balancer_class,
            load_balancer_ip=load_balancer_ip,
            load_balancer_source_ranges=load_balancer_source_ranges,
            port=port,
            publish_not_ready_addresses=publish_not_ready_addresses,
            selector=selector,
            session_affinity=session_affinity,
            session_affinity_config=session_affinity_config,
            type=type,
        )

        return typing.cast(None, jsii.invoke(self, "putSpec", [value]))

    @jsii.member(jsii_name="putTimeouts")
    def put_timeouts(self, *, create: typing.Optional[builtins.str] = None) -> None:
        '''
        :param create: Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#create Service#create}.
        '''
        value = ServiceTimeouts(create=create)

        return typing.cast(None, jsii.invoke(self, "putTimeouts", [value]))

    @jsii.member(jsii_name="resetId")
    def reset_id(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetId", []))

    @jsii.member(jsii_name="resetTimeouts")
    def reset_timeouts(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetTimeouts", []))

    @jsii.member(jsii_name="resetWaitForLoadBalancer")
    def reset_wait_for_load_balancer(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetWaitForLoadBalancer", []))

    @jsii.member(jsii_name="synthesizeAttributes")
    def _synthesize_attributes(self) -> typing.Mapping[builtins.str, typing.Any]:
        return typing.cast(typing.Mapping[builtins.str, typing.Any], jsii.invoke(self, "synthesizeAttributes", []))

    @jsii.python.classproperty
    @jsii.member(jsii_name="tfResourceType")
    def TF_RESOURCE_TYPE(cls) -> builtins.str:
        return typing.cast(builtins.str, jsii.sget(cls, "tfResourceType"))

    @builtins.property
    @jsii.member(jsii_name="metadata")
    def metadata(self) -> "ServiceMetadataOutputReference":
        return typing.cast("ServiceMetadataOutputReference", jsii.get(self, "metadata"))

    @builtins.property
    @jsii.member(jsii_name="spec")
    def spec(self) -> "ServiceSpecOutputReference":
        return typing.cast("ServiceSpecOutputReference", jsii.get(self, "spec"))

    @builtins.property
    @jsii.member(jsii_name="status")
    def status(self) -> "ServiceStatusList":
        return typing.cast("ServiceStatusList", jsii.get(self, "status"))

    @builtins.property
    @jsii.member(jsii_name="timeouts")
    def timeouts(self) -> "ServiceTimeoutsOutputReference":
        return typing.cast("ServiceTimeoutsOutputReference", jsii.get(self, "timeouts"))

    @builtins.property
    @jsii.member(jsii_name="idInput")
    def id_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "idInput"))

    @builtins.property
    @jsii.member(jsii_name="metadataInput")
    def metadata_input(self) -> typing.Optional["ServiceMetadata"]:
        return typing.cast(typing.Optional["ServiceMetadata"], jsii.get(self, "metadataInput"))

    @builtins.property
    @jsii.member(jsii_name="specInput")
    def spec_input(self) -> typing.Optional["ServiceSpec"]:
        return typing.cast(typing.Optional["ServiceSpec"], jsii.get(self, "specInput"))

    @builtins.property
    @jsii.member(jsii_name="timeoutsInput")
    def timeouts_input(
        self,
    ) -> typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, "ServiceTimeouts"]]:
        return typing.cast(typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, "ServiceTimeouts"]], jsii.get(self, "timeoutsInput"))

    @builtins.property
    @jsii.member(jsii_name="waitForLoadBalancerInput")
    def wait_for_load_balancer_input(
        self,
    ) -> typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]]:
        return typing.cast(typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]], jsii.get(self, "waitForLoadBalancerInput"))

    @builtins.property
    @jsii.member(jsii_name="id")
    def id(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "id"))

    @id.setter
    def id(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__e7022b3079da14277606eaf8ed8a1a3d324bc22bc24e8c5416be6eaf49bd5dd4)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "id", value)

    @builtins.property
    @jsii.member(jsii_name="waitForLoadBalancer")
    def wait_for_load_balancer(
        self,
    ) -> typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]:
        return typing.cast(typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable], jsii.get(self, "waitForLoadBalancer"))

    @wait_for_load_balancer.setter
    def wait_for_load_balancer(
        self,
        value: typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable],
    ) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__80cb12f70e7be78328ea4c478ae9dc647443f48b01a34bf654ed2d9a59d00c14)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "waitForLoadBalancer", value)


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceConfig",
    jsii_struct_bases=[_cdktf_9a9027ec.TerraformMetaArguments],
    name_mapping={
        "connection": "connection",
        "count": "count",
        "depends_on": "dependsOn",
        "for_each": "forEach",
        "lifecycle": "lifecycle",
        "provider": "provider",
        "provisioners": "provisioners",
        "metadata": "metadata",
        "spec": "spec",
        "id": "id",
        "timeouts": "timeouts",
        "wait_for_load_balancer": "waitForLoadBalancer",
    },
)
class ServiceConfig(_cdktf_9a9027ec.TerraformMetaArguments):
    def __init__(
        self,
        *,
        connection: typing.Optional[typing.Union[typing.Union[_cdktf_9a9027ec.SSHProvisionerConnection, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.WinrmProvisionerConnection, typing.Dict[builtins.str, typing.Any]]]] = None,
        count: typing.Optional[typing.Union[jsii.Number, _cdktf_9a9027ec.TerraformCount]] = None,
        depends_on: typing.Optional[typing.Sequence[_cdktf_9a9027ec.ITerraformDependable]] = None,
        for_each: typing.Optional[_cdktf_9a9027ec.ITerraformIterator] = None,
        lifecycle: typing.Optional[typing.Union[_cdktf_9a9027ec.TerraformResourceLifecycle, typing.Dict[builtins.str, typing.Any]]] = None,
        provider: typing.Optional[_cdktf_9a9027ec.TerraformProvider] = None,
        provisioners: typing.Optional[typing.Sequence[typing.Union[typing.Union[_cdktf_9a9027ec.FileProvisioner, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.LocalExecProvisioner, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.RemoteExecProvisioner, typing.Dict[builtins.str, typing.Any]]]]] = None,
        metadata: typing.Union["ServiceMetadata", typing.Dict[builtins.str, typing.Any]],
        spec: typing.Union["ServiceSpec", typing.Dict[builtins.str, typing.Any]],
        id: typing.Optional[builtins.str] = None,
        timeouts: typing.Optional[typing.Union["ServiceTimeouts", typing.Dict[builtins.str, typing.Any]]] = None,
        wait_for_load_balancer: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
    ) -> None:
        '''
        :param connection: 
        :param count: 
        :param depends_on: 
        :param for_each: 
        :param lifecycle: 
        :param provider: 
        :param provisioners: 
        :param metadata: metadata block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#metadata Service#metadata}
        :param spec: spec block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#spec Service#spec}
        :param id: Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#id Service#id}. Please be aware that the id field is automatically added to all resources in Terraform providers using a Terraform provider SDK version below 2. If you experience problems setting this value it might not be settable. Please take a look at the provider documentation to ensure it should be settable.
        :param timeouts: timeouts block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#timeouts Service#timeouts}
        :param wait_for_load_balancer: Terraform will wait for the load balancer to have at least 1 endpoint before considering the resource created. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#wait_for_load_balancer Service#wait_for_load_balancer}
        '''
        if isinstance(lifecycle, dict):
            lifecycle = _cdktf_9a9027ec.TerraformResourceLifecycle(**lifecycle)
        if isinstance(metadata, dict):
            metadata = ServiceMetadata(**metadata)
        if isinstance(spec, dict):
            spec = ServiceSpec(**spec)
        if isinstance(timeouts, dict):
            timeouts = ServiceTimeouts(**timeouts)
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__5705bb667ac158d1879c1a562c6e3206deb51c5f25f09b6f0d1223e19deef331)
            check_type(argname="argument connection", value=connection, expected_type=type_hints["connection"])
            check_type(argname="argument count", value=count, expected_type=type_hints["count"])
            check_type(argname="argument depends_on", value=depends_on, expected_type=type_hints["depends_on"])
            check_type(argname="argument for_each", value=for_each, expected_type=type_hints["for_each"])
            check_type(argname="argument lifecycle", value=lifecycle, expected_type=type_hints["lifecycle"])
            check_type(argname="argument provider", value=provider, expected_type=type_hints["provider"])
            check_type(argname="argument provisioners", value=provisioners, expected_type=type_hints["provisioners"])
            check_type(argname="argument metadata", value=metadata, expected_type=type_hints["metadata"])
            check_type(argname="argument spec", value=spec, expected_type=type_hints["spec"])
            check_type(argname="argument id", value=id, expected_type=type_hints["id"])
            check_type(argname="argument timeouts", value=timeouts, expected_type=type_hints["timeouts"])
            check_type(argname="argument wait_for_load_balancer", value=wait_for_load_balancer, expected_type=type_hints["wait_for_load_balancer"])
        self._values: typing.Dict[builtins.str, typing.Any] = {
            "metadata": metadata,
            "spec": spec,
        }
        if connection is not None:
            self._values["connection"] = connection
        if count is not None:
            self._values["count"] = count
        if depends_on is not None:
            self._values["depends_on"] = depends_on
        if for_each is not None:
            self._values["for_each"] = for_each
        if lifecycle is not None:
            self._values["lifecycle"] = lifecycle
        if provider is not None:
            self._values["provider"] = provider
        if provisioners is not None:
            self._values["provisioners"] = provisioners
        if id is not None:
            self._values["id"] = id
        if timeouts is not None:
            self._values["timeouts"] = timeouts
        if wait_for_load_balancer is not None:
            self._values["wait_for_load_balancer"] = wait_for_load_balancer

    @builtins.property
    def connection(
        self,
    ) -> typing.Optional[typing.Union[_cdktf_9a9027ec.SSHProvisionerConnection, _cdktf_9a9027ec.WinrmProvisionerConnection]]:
        '''
        :stability: experimental
        '''
        result = self._values.get("connection")
        return typing.cast(typing.Optional[typing.Union[_cdktf_9a9027ec.SSHProvisionerConnection, _cdktf_9a9027ec.WinrmProvisionerConnection]], result)

    @builtins.property
    def count(
        self,
    ) -> typing.Optional[typing.Union[jsii.Number, _cdktf_9a9027ec.TerraformCount]]:
        '''
        :stability: experimental
        '''
        result = self._values.get("count")
        return typing.cast(typing.Optional[typing.Union[jsii.Number, _cdktf_9a9027ec.TerraformCount]], result)

    @builtins.property
    def depends_on(
        self,
    ) -> typing.Optional[typing.List[_cdktf_9a9027ec.ITerraformDependable]]:
        '''
        :stability: experimental
        '''
        result = self._values.get("depends_on")
        return typing.cast(typing.Optional[typing.List[_cdktf_9a9027ec.ITerraformDependable]], result)

    @builtins.property
    def for_each(self) -> typing.Optional[_cdktf_9a9027ec.ITerraformIterator]:
        '''
        :stability: experimental
        '''
        result = self._values.get("for_each")
        return typing.cast(typing.Optional[_cdktf_9a9027ec.ITerraformIterator], result)

    @builtins.property
    def lifecycle(self) -> typing.Optional[_cdktf_9a9027ec.TerraformResourceLifecycle]:
        '''
        :stability: experimental
        '''
        result = self._values.get("lifecycle")
        return typing.cast(typing.Optional[_cdktf_9a9027ec.TerraformResourceLifecycle], result)

    @builtins.property
    def provider(self) -> typing.Optional[_cdktf_9a9027ec.TerraformProvider]:
        '''
        :stability: experimental
        '''
        result = self._values.get("provider")
        return typing.cast(typing.Optional[_cdktf_9a9027ec.TerraformProvider], result)

    @builtins.property
    def provisioners(
        self,
    ) -> typing.Optional[typing.List[typing.Union[_cdktf_9a9027ec.FileProvisioner, _cdktf_9a9027ec.LocalExecProvisioner, _cdktf_9a9027ec.RemoteExecProvisioner]]]:
        '''
        :stability: experimental
        '''
        result = self._values.get("provisioners")
        return typing.cast(typing.Optional[typing.List[typing.Union[_cdktf_9a9027ec.FileProvisioner, _cdktf_9a9027ec.LocalExecProvisioner, _cdktf_9a9027ec.RemoteExecProvisioner]]], result)

    @builtins.property
    def metadata(self) -> "ServiceMetadata":
        '''metadata block.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#metadata Service#metadata}
        '''
        result = self._values.get("metadata")
        assert result is not None, "Required property 'metadata' is missing"
        return typing.cast("ServiceMetadata", result)

    @builtins.property
    def spec(self) -> "ServiceSpec":
        '''spec block.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#spec Service#spec}
        '''
        result = self._values.get("spec")
        assert result is not None, "Required property 'spec' is missing"
        return typing.cast("ServiceSpec", result)

    @builtins.property
    def id(self) -> typing.Optional[builtins.str]:
        '''Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#id Service#id}.

        Please be aware that the id field is automatically added to all resources in Terraform providers using a Terraform provider SDK version below 2.
        If you experience problems setting this value it might not be settable. Please take a look at the provider documentation to ensure it should be settable.
        '''
        result = self._values.get("id")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def timeouts(self) -> typing.Optional["ServiceTimeouts"]:
        '''timeouts block.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#timeouts Service#timeouts}
        '''
        result = self._values.get("timeouts")
        return typing.cast(typing.Optional["ServiceTimeouts"], result)

    @builtins.property
    def wait_for_load_balancer(
        self,
    ) -> typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]]:
        '''Terraform will wait for the load balancer to have at least 1 endpoint before considering the resource created.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#wait_for_load_balancer Service#wait_for_load_balancer}
        '''
        result = self._values.get("wait_for_load_balancer")
        return typing.cast(typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]], result)

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceConfig(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceMetadata",
    jsii_struct_bases=[],
    name_mapping={
        "annotations": "annotations",
        "generate_name": "generateName",
        "labels": "labels",
        "name": "name",
        "namespace": "namespace",
    },
)
class ServiceMetadata:
    def __init__(
        self,
        *,
        annotations: typing.Optional[typing.Mapping[builtins.str, builtins.str]] = None,
        generate_name: typing.Optional[builtins.str] = None,
        labels: typing.Optional[typing.Mapping[builtins.str, builtins.str]] = None,
        name: typing.Optional[builtins.str] = None,
        namespace: typing.Optional[builtins.str] = None,
    ) -> None:
        '''
        :param annotations: An unstructured key value map stored with the service that may be used to store arbitrary metadata. More info: http://kubernetes.io/docs/user-guide/annotations Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#annotations Service#annotations}
        :param generate_name: Prefix, used by the server, to generate a unique name ONLY IF the ``name`` field has not been provided. This value will also be combined with a unique suffix. Read more: https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#idempotency Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#generate_name Service#generate_name}
        :param labels: Map of string keys and values that can be used to organize and categorize (scope and select) the service. May match selectors of replication controllers and services. More info: http://kubernetes.io/docs/user-guide/labels Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#labels Service#labels}
        :param name: Name of the service, must be unique. Cannot be updated. More info: http://kubernetes.io/docs/user-guide/identifiers#names. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#name Service#name}
        :param namespace: Namespace defines the space within which name of the service must be unique. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#namespace Service#namespace}
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__f83ac22dc025dff9cab6cf84ee3b09e78e5651e681b883d00a283cfb04e13a91)
            check_type(argname="argument annotations", value=annotations, expected_type=type_hints["annotations"])
            check_type(argname="argument generate_name", value=generate_name, expected_type=type_hints["generate_name"])
            check_type(argname="argument labels", value=labels, expected_type=type_hints["labels"])
            check_type(argname="argument name", value=name, expected_type=type_hints["name"])
            check_type(argname="argument namespace", value=namespace, expected_type=type_hints["namespace"])
        self._values: typing.Dict[builtins.str, typing.Any] = {}
        if annotations is not None:
            self._values["annotations"] = annotations
        if generate_name is not None:
            self._values["generate_name"] = generate_name
        if labels is not None:
            self._values["labels"] = labels
        if name is not None:
            self._values["name"] = name
        if namespace is not None:
            self._values["namespace"] = namespace

    @builtins.property
    def annotations(
        self,
    ) -> typing.Optional[typing.Mapping[builtins.str, builtins.str]]:
        '''An unstructured key value map stored with the service that may be used to store arbitrary metadata.

        More info: http://kubernetes.io/docs/user-guide/annotations

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#annotations Service#annotations}
        '''
        result = self._values.get("annotations")
        return typing.cast(typing.Optional[typing.Mapping[builtins.str, builtins.str]], result)

    @builtins.property
    def generate_name(self) -> typing.Optional[builtins.str]:
        '''Prefix, used by the server, to generate a unique name ONLY IF the ``name`` field has not been provided.

        This value will also be combined with a unique suffix. Read more: https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#idempotency

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#generate_name Service#generate_name}
        '''
        result = self._values.get("generate_name")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def labels(self) -> typing.Optional[typing.Mapping[builtins.str, builtins.str]]:
        '''Map of string keys and values that can be used to organize and categorize (scope and select) the service.

        May match selectors of replication controllers and services. More info: http://kubernetes.io/docs/user-guide/labels

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#labels Service#labels}
        '''
        result = self._values.get("labels")
        return typing.cast(typing.Optional[typing.Mapping[builtins.str, builtins.str]], result)

    @builtins.property
    def name(self) -> typing.Optional[builtins.str]:
        '''Name of the service, must be unique. Cannot be updated. More info: http://kubernetes.io/docs/user-guide/identifiers#names.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#name Service#name}
        '''
        result = self._values.get("name")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def namespace(self) -> typing.Optional[builtins.str]:
        '''Namespace defines the space within which name of the service must be unique.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#namespace Service#namespace}
        '''
        result = self._values.get("namespace")
        return typing.cast(typing.Optional[builtins.str], result)

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceMetadata(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


class ServiceMetadataOutputReference(
    _cdktf_9a9027ec.ComplexObject,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceMetadataOutputReference",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__2272ee7a402d6118de95eb47fb9d04a3ed21a1dbdfd036db7acafe4bd4e4c0b9)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute])

    @jsii.member(jsii_name="resetAnnotations")
    def reset_annotations(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetAnnotations", []))

    @jsii.member(jsii_name="resetGenerateName")
    def reset_generate_name(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetGenerateName", []))

    @jsii.member(jsii_name="resetLabels")
    def reset_labels(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetLabels", []))

    @jsii.member(jsii_name="resetName")
    def reset_name(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetName", []))

    @jsii.member(jsii_name="resetNamespace")
    def reset_namespace(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetNamespace", []))

    @builtins.property
    @jsii.member(jsii_name="generation")
    def generation(self) -> jsii.Number:
        return typing.cast(jsii.Number, jsii.get(self, "generation"))

    @builtins.property
    @jsii.member(jsii_name="resourceVersion")
    def resource_version(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "resourceVersion"))

    @builtins.property
    @jsii.member(jsii_name="uid")
    def uid(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "uid"))

    @builtins.property
    @jsii.member(jsii_name="annotationsInput")
    def annotations_input(
        self,
    ) -> typing.Optional[typing.Mapping[builtins.str, builtins.str]]:
        return typing.cast(typing.Optional[typing.Mapping[builtins.str, builtins.str]], jsii.get(self, "annotationsInput"))

    @builtins.property
    @jsii.member(jsii_name="generateNameInput")
    def generate_name_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "generateNameInput"))

    @builtins.property
    @jsii.member(jsii_name="labelsInput")
    def labels_input(
        self,
    ) -> typing.Optional[typing.Mapping[builtins.str, builtins.str]]:
        return typing.cast(typing.Optional[typing.Mapping[builtins.str, builtins.str]], jsii.get(self, "labelsInput"))

    @builtins.property
    @jsii.member(jsii_name="nameInput")
    def name_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "nameInput"))

    @builtins.property
    @jsii.member(jsii_name="namespaceInput")
    def namespace_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "namespaceInput"))

    @builtins.property
    @jsii.member(jsii_name="annotations")
    def annotations(self) -> typing.Mapping[builtins.str, builtins.str]:
        return typing.cast(typing.Mapping[builtins.str, builtins.str], jsii.get(self, "annotations"))

    @annotations.setter
    def annotations(self, value: typing.Mapping[builtins.str, builtins.str]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__10d98e551c5a15e59e1f1b47b622816536e176c8724d89d3fbf62c34ab64e283)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "annotations", value)

    @builtins.property
    @jsii.member(jsii_name="generateName")
    def generate_name(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "generateName"))

    @generate_name.setter
    def generate_name(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__9c4ea2d9c52da51e832a122907ba3bb96ce3b3fea4419b44d80b3f8bbaf42d9e)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "generateName", value)

    @builtins.property
    @jsii.member(jsii_name="labels")
    def labels(self) -> typing.Mapping[builtins.str, builtins.str]:
        return typing.cast(typing.Mapping[builtins.str, builtins.str], jsii.get(self, "labels"))

    @labels.setter
    def labels(self, value: typing.Mapping[builtins.str, builtins.str]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__12b452541e5d3b1e3bca70f73f121afced3665878ee15e179799346351aefff2)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "labels", value)

    @builtins.property
    @jsii.member(jsii_name="name")
    def name(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "name"))

    @name.setter
    def name(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__699c745522b77730f7efd0081478c62be8f9b66ea42a06c900385149b4e1d72e)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "name", value)

    @builtins.property
    @jsii.member(jsii_name="namespace")
    def namespace(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "namespace"))

    @namespace.setter
    def namespace(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__15ae42c0c38ee4c7b68c1ed04bcfa74e64b8c1bb1c780a5f07a3b9a4a8eed5e5)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "namespace", value)

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(self) -> typing.Optional[ServiceMetadata]:
        return typing.cast(typing.Optional[ServiceMetadata], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(self, value: typing.Optional[ServiceMetadata]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__e6f010dd962f101be9896adf9347fba896da5bea0341b0852ccda31380df9685)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceSpec",
    jsii_struct_bases=[],
    name_mapping={
        "allocate_load_balancer_node_ports": "allocateLoadBalancerNodePorts",
        "cluster_ip": "clusterIp",
        "cluster_ips": "clusterIps",
        "external_ips": "externalIps",
        "external_name": "externalName",
        "external_traffic_policy": "externalTrafficPolicy",
        "health_check_node_port": "healthCheckNodePort",
        "internal_traffic_policy": "internalTrafficPolicy",
        "ip_families": "ipFamilies",
        "ip_family_policy": "ipFamilyPolicy",
        "load_balancer_class": "loadBalancerClass",
        "load_balancer_ip": "loadBalancerIp",
        "load_balancer_source_ranges": "loadBalancerSourceRanges",
        "port": "port",
        "publish_not_ready_addresses": "publishNotReadyAddresses",
        "selector": "selector",
        "session_affinity": "sessionAffinity",
        "session_affinity_config": "sessionAffinityConfig",
        "type": "type",
    },
)
class ServiceSpec:
    def __init__(
        self,
        *,
        allocate_load_balancer_node_ports: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
        cluster_ip: typing.Optional[builtins.str] = None,
        cluster_ips: typing.Optional[typing.Sequence[builtins.str]] = None,
        external_ips: typing.Optional[typing.Sequence[builtins.str]] = None,
        external_name: typing.Optional[builtins.str] = None,
        external_traffic_policy: typing.Optional[builtins.str] = None,
        health_check_node_port: typing.Optional[jsii.Number] = None,
        internal_traffic_policy: typing.Optional[builtins.str] = None,
        ip_families: typing.Optional[typing.Sequence[builtins.str]] = None,
        ip_family_policy: typing.Optional[builtins.str] = None,
        load_balancer_class: typing.Optional[builtins.str] = None,
        load_balancer_ip: typing.Optional[builtins.str] = None,
        load_balancer_source_ranges: typing.Optional[typing.Sequence[builtins.str]] = None,
        port: typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.Sequence[typing.Union["ServiceSpecPort", typing.Dict[builtins.str, typing.Any]]]]] = None,
        publish_not_ready_addresses: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
        selector: typing.Optional[typing.Mapping[builtins.str, builtins.str]] = None,
        session_affinity: typing.Optional[builtins.str] = None,
        session_affinity_config: typing.Optional[typing.Union["ServiceSpecSessionAffinityConfig", typing.Dict[builtins.str, typing.Any]]] = None,
        type: typing.Optional[builtins.str] = None,
    ) -> None:
        '''
        :param allocate_load_balancer_node_ports: Defines if ``NodePorts`` will be automatically allocated for services with type ``LoadBalancer``. It may be set to ``false`` if the cluster load-balancer does not rely on ``NodePorts``. If the caller requests specific ``NodePorts`` (by specifying a value), those requests will be respected, regardless of this field. This field may only be set for services with type ``LoadBalancer``. Default is ``true``. More info: https://kubernetes.io/docs/concepts/services-networking/service/#load-balancer-nodeport-allocation Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#allocate_load_balancer_node_ports Service#allocate_load_balancer_node_ports}
        :param cluster_ip: The IP address of the service. It is usually assigned randomly by the master. If an address is specified manually and is not in use by others, it will be allocated to the service; otherwise, creation of the service will fail. ``None`` can be specified for headless services when proxying is not required. Ignored if type is ``ExternalName``. More info: http://kubernetes.io/docs/user-guide/services#virtual-ips-and-service-proxies Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#cluster_ip Service#cluster_ip}
        :param cluster_ips: List of IP addresses assigned to this service, and are usually assigned randomly. If an address is specified manually and is not in use by others, it will be allocated to the service; otherwise creation of the service will fail. If this field is not specified, it will be initialized from the ``clusterIP`` field. If this field is specified, clients must ensure that ``clusterIPs[0]`` and ``clusterIP`` have the same value. More info: http://kubernetes.io/docs/user-guide/services#virtual-ips-and-service-proxies Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#cluster_ips Service#cluster_ips}
        :param external_ips: A list of IP addresses for which nodes in the cluster will also accept traffic for this service. These IPs are not managed by Kubernetes. The user is responsible for ensuring that traffic arrives at a node with this IP. A common example is external load-balancers that are not part of the Kubernetes system. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#external_ips Service#external_ips}
        :param external_name: The external reference that kubedns or equivalent will return as a CNAME record for this service. No proxying will be involved. Must be a valid DNS name and requires ``type`` to be ``ExternalName``. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#external_name Service#external_name}
        :param external_traffic_policy: Denotes if this Service desires to route external traffic to node-local or cluster-wide endpoints. ``Local`` preserves the client source IP and avoids a second hop for LoadBalancer and Nodeport type services, but risks potentially imbalanced traffic spreading. ``Cluster`` obscures the client source IP and may cause a second hop to another node, but should have good overall load-spreading. More info: https://kubernetes.io/docs/tutorials/services/source-ip/ Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#external_traffic_policy Service#external_traffic_policy}
        :param health_check_node_port: Specifies the Healthcheck NodePort for the service. Only effects when type is set to ``LoadBalancer`` and external_traffic_policy is set to ``Local``. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#health_check_node_port Service#health_check_node_port}
        :param internal_traffic_policy: Specifies if the cluster internal traffic should be routed to all endpoints or node-local endpoints only. ``Cluster`` routes internal traffic to a Service to all endpoints. ``Local`` routes traffic to node-local endpoints only, traffic is dropped if no node-local endpoints are ready. The default value is ``Cluster``. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#internal_traffic_policy Service#internal_traffic_policy}
        :param ip_families: IPFamilies is a list of IP families (e.g. IPv4, IPv6) assigned to this service. This field is usually assigned automatically based on cluster configuration and the ipFamilyPolicy field. If this field is specified manually, the requested family is available in the cluster, and ipFamilyPolicy allows it, it will be used; otherwise creation of the service will fail. This field is conditionally mutable: it allows for adding or removing a secondary IP family, but it does not allow changing the primary IP family of the Service. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#ip_families Service#ip_families}
        :param ip_family_policy: IPFamilyPolicy represents the dual-stack-ness requested or required by this Service. If there is no value provided, then this field will be set to SingleStack. Services can be 'SingleStack' (a single IP family), 'PreferDualStack' (two IP families on dual-stack configured clusters or a single IP family on single-stack clusters), or 'RequireDualStack' (two IP families on dual-stack configured clusters, otherwise fail). The ipFamilies and clusterIPs fields depend on the value of this field. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#ip_family_policy Service#ip_family_policy}
        :param load_balancer_class: The class of the load balancer implementation this Service belongs to. If specified, the value of this field must be a label-style identifier, with an optional prefix. This field can only be set when the Service type is ``LoadBalancer``. If not set, the default load balancer implementation is used. This field can only be set when creating or updating a Service to type ``LoadBalancer``. More info: https://kubernetes.io/docs/concepts/services-networking/service/#load-balancer-class Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#load_balancer_class Service#load_balancer_class}
        :param load_balancer_ip: Only applies to ``type = LoadBalancer``. LoadBalancer will get created with the IP specified in this field. This feature depends on whether the underlying cloud-provider supports specifying this field when a load balancer is created. This field will be ignored if the cloud-provider does not support the feature. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#load_balancer_ip Service#load_balancer_ip}
        :param load_balancer_source_ranges: If specified and supported by the platform, this will restrict traffic through the cloud-provider load-balancer will be restricted to the specified client IPs. This field will be ignored if the cloud-provider does not support the feature. More info: http://kubernetes.io/docs/user-guide/services-firewalls Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#load_balancer_source_ranges Service#load_balancer_source_ranges}
        :param port: port block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#port Service#port}
        :param publish_not_ready_addresses: When set to true, indicates that DNS implementations must publish the ``notReadyAddresses`` of subsets for the Endpoints associated with the Service. The default value is ``false``. The primary use case for setting this field is to use a StatefulSet's Headless Service to propagate ``SRV`` records for its Pods without respect to their readiness for purpose of peer discovery. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#publish_not_ready_addresses Service#publish_not_ready_addresses}
        :param selector: Route service traffic to pods with label keys and values matching this selector. Only applies to types ``ClusterIP``, ``NodePort``, and ``LoadBalancer``. More info: http://kubernetes.io/docs/user-guide/services#overview Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#selector Service#selector}
        :param session_affinity: Used to maintain session affinity. Supports ``ClientIP`` and ``None``. Defaults to ``None``. More info: http://kubernetes.io/docs/user-guide/services#virtual-ips-and-service-proxies. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#session_affinity Service#session_affinity}
        :param session_affinity_config: session_affinity_config block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#session_affinity_config Service#session_affinity_config}
        :param type: Determines how the service is exposed. Defaults to ``ClusterIP``. Valid options are ``ExternalName``, ``ClusterIP``, ``NodePort``, and ``LoadBalancer``. ``ExternalName`` maps to the specified ``external_name``. More info: http://kubernetes.io/docs/user-guide/services#overview Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#type Service#type}
        '''
        if isinstance(session_affinity_config, dict):
            session_affinity_config = ServiceSpecSessionAffinityConfig(**session_affinity_config)
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__03f320219ee5eea7af4f1ae3a7dbc7c2815442cfb1407e89cc3939597b51cb94)
            check_type(argname="argument allocate_load_balancer_node_ports", value=allocate_load_balancer_node_ports, expected_type=type_hints["allocate_load_balancer_node_ports"])
            check_type(argname="argument cluster_ip", value=cluster_ip, expected_type=type_hints["cluster_ip"])
            check_type(argname="argument cluster_ips", value=cluster_ips, expected_type=type_hints["cluster_ips"])
            check_type(argname="argument external_ips", value=external_ips, expected_type=type_hints["external_ips"])
            check_type(argname="argument external_name", value=external_name, expected_type=type_hints["external_name"])
            check_type(argname="argument external_traffic_policy", value=external_traffic_policy, expected_type=type_hints["external_traffic_policy"])
            check_type(argname="argument health_check_node_port", value=health_check_node_port, expected_type=type_hints["health_check_node_port"])
            check_type(argname="argument internal_traffic_policy", value=internal_traffic_policy, expected_type=type_hints["internal_traffic_policy"])
            check_type(argname="argument ip_families", value=ip_families, expected_type=type_hints["ip_families"])
            check_type(argname="argument ip_family_policy", value=ip_family_policy, expected_type=type_hints["ip_family_policy"])
            check_type(argname="argument load_balancer_class", value=load_balancer_class, expected_type=type_hints["load_balancer_class"])
            check_type(argname="argument load_balancer_ip", value=load_balancer_ip, expected_type=type_hints["load_balancer_ip"])
            check_type(argname="argument load_balancer_source_ranges", value=load_balancer_source_ranges, expected_type=type_hints["load_balancer_source_ranges"])
            check_type(argname="argument port", value=port, expected_type=type_hints["port"])
            check_type(argname="argument publish_not_ready_addresses", value=publish_not_ready_addresses, expected_type=type_hints["publish_not_ready_addresses"])
            check_type(argname="argument selector", value=selector, expected_type=type_hints["selector"])
            check_type(argname="argument session_affinity", value=session_affinity, expected_type=type_hints["session_affinity"])
            check_type(argname="argument session_affinity_config", value=session_affinity_config, expected_type=type_hints["session_affinity_config"])
            check_type(argname="argument type", value=type, expected_type=type_hints["type"])
        self._values: typing.Dict[builtins.str, typing.Any] = {}
        if allocate_load_balancer_node_ports is not None:
            self._values["allocate_load_balancer_node_ports"] = allocate_load_balancer_node_ports
        if cluster_ip is not None:
            self._values["cluster_ip"] = cluster_ip
        if cluster_ips is not None:
            self._values["cluster_ips"] = cluster_ips
        if external_ips is not None:
            self._values["external_ips"] = external_ips
        if external_name is not None:
            self._values["external_name"] = external_name
        if external_traffic_policy is not None:
            self._values["external_traffic_policy"] = external_traffic_policy
        if health_check_node_port is not None:
            self._values["health_check_node_port"] = health_check_node_port
        if internal_traffic_policy is not None:
            self._values["internal_traffic_policy"] = internal_traffic_policy
        if ip_families is not None:
            self._values["ip_families"] = ip_families
        if ip_family_policy is not None:
            self._values["ip_family_policy"] = ip_family_policy
        if load_balancer_class is not None:
            self._values["load_balancer_class"] = load_balancer_class
        if load_balancer_ip is not None:
            self._values["load_balancer_ip"] = load_balancer_ip
        if load_balancer_source_ranges is not None:
            self._values["load_balancer_source_ranges"] = load_balancer_source_ranges
        if port is not None:
            self._values["port"] = port
        if publish_not_ready_addresses is not None:
            self._values["publish_not_ready_addresses"] = publish_not_ready_addresses
        if selector is not None:
            self._values["selector"] = selector
        if session_affinity is not None:
            self._values["session_affinity"] = session_affinity
        if session_affinity_config is not None:
            self._values["session_affinity_config"] = session_affinity_config
        if type is not None:
            self._values["type"] = type

    @builtins.property
    def allocate_load_balancer_node_ports(
        self,
    ) -> typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]]:
        '''Defines if ``NodePorts`` will be automatically allocated for services with type ``LoadBalancer``.

        It may be set to ``false`` if the cluster load-balancer does not rely on ``NodePorts``.  If the caller requests specific ``NodePorts`` (by specifying a value), those requests will be respected, regardless of this field. This field may only be set for services with type ``LoadBalancer``. Default is ``true``. More info: https://kubernetes.io/docs/concepts/services-networking/service/#load-balancer-nodeport-allocation

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#allocate_load_balancer_node_ports Service#allocate_load_balancer_node_ports}
        '''
        result = self._values.get("allocate_load_balancer_node_ports")
        return typing.cast(typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]], result)

    @builtins.property
    def cluster_ip(self) -> typing.Optional[builtins.str]:
        '''The IP address of the service.

        It is usually assigned randomly by the master. If an address is specified manually and is not in use by others, it will be allocated to the service; otherwise, creation of the service will fail. ``None`` can be specified for headless services when proxying is not required. Ignored if type is ``ExternalName``. More info: http://kubernetes.io/docs/user-guide/services#virtual-ips-and-service-proxies

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#cluster_ip Service#cluster_ip}
        '''
        result = self._values.get("cluster_ip")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def cluster_ips(self) -> typing.Optional[typing.List[builtins.str]]:
        '''List of IP addresses assigned to this service, and are usually assigned randomly.

        If an address is specified manually and is not in use by others, it will be allocated to the service; otherwise creation of the service will fail. If this field is not specified, it will be initialized from the ``clusterIP`` field. If this field is specified, clients must ensure that ``clusterIPs[0]`` and ``clusterIP`` have the same value. More info: http://kubernetes.io/docs/user-guide/services#virtual-ips-and-service-proxies

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#cluster_ips Service#cluster_ips}
        '''
        result = self._values.get("cluster_ips")
        return typing.cast(typing.Optional[typing.List[builtins.str]], result)

    @builtins.property
    def external_ips(self) -> typing.Optional[typing.List[builtins.str]]:
        '''A list of IP addresses for which nodes in the cluster will also accept traffic for this service.

        These IPs are not managed by Kubernetes. The user is responsible for ensuring that traffic arrives at a node with this IP.  A common example is external load-balancers that are not part of the Kubernetes system.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#external_ips Service#external_ips}
        '''
        result = self._values.get("external_ips")
        return typing.cast(typing.Optional[typing.List[builtins.str]], result)

    @builtins.property
    def external_name(self) -> typing.Optional[builtins.str]:
        '''The external reference that kubedns or equivalent will return as a CNAME record for this service.

        No proxying will be involved. Must be a valid DNS name and requires ``type`` to be ``ExternalName``.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#external_name Service#external_name}
        '''
        result = self._values.get("external_name")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def external_traffic_policy(self) -> typing.Optional[builtins.str]:
        '''Denotes if this Service desires to route external traffic to node-local or cluster-wide endpoints.

        ``Local`` preserves the client source IP and avoids a second hop for LoadBalancer and Nodeport type services, but risks potentially imbalanced traffic spreading. ``Cluster`` obscures the client source IP and may cause a second hop to another node, but should have good overall load-spreading. More info: https://kubernetes.io/docs/tutorials/services/source-ip/

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#external_traffic_policy Service#external_traffic_policy}
        '''
        result = self._values.get("external_traffic_policy")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def health_check_node_port(self) -> typing.Optional[jsii.Number]:
        '''Specifies the Healthcheck NodePort for the service.

        Only effects when type is set to ``LoadBalancer`` and external_traffic_policy is set to ``Local``.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#health_check_node_port Service#health_check_node_port}
        '''
        result = self._values.get("health_check_node_port")
        return typing.cast(typing.Optional[jsii.Number], result)

    @builtins.property
    def internal_traffic_policy(self) -> typing.Optional[builtins.str]:
        '''Specifies if the cluster internal traffic should be routed to all endpoints or node-local endpoints only.

        ``Cluster`` routes internal traffic to a Service to all endpoints. ``Local`` routes traffic to node-local endpoints only, traffic is dropped if no node-local endpoints are ready. The default value is ``Cluster``.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#internal_traffic_policy Service#internal_traffic_policy}
        '''
        result = self._values.get("internal_traffic_policy")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def ip_families(self) -> typing.Optional[typing.List[builtins.str]]:
        '''IPFamilies is a list of IP families (e.g. IPv4, IPv6) assigned to this service. This field is usually assigned automatically based on cluster configuration and the ipFamilyPolicy field. If this field is specified manually, the requested family is available in the cluster, and ipFamilyPolicy allows it, it will be used; otherwise creation of the service will fail. This field is conditionally mutable: it allows for adding or removing a secondary IP family, but it does not allow changing the primary IP family of the Service.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#ip_families Service#ip_families}
        '''
        result = self._values.get("ip_families")
        return typing.cast(typing.Optional[typing.List[builtins.str]], result)

    @builtins.property
    def ip_family_policy(self) -> typing.Optional[builtins.str]:
        '''IPFamilyPolicy represents the dual-stack-ness requested or required by this Service.

        If there is no value provided, then this field will be set to SingleStack. Services can be 'SingleStack' (a single IP family), 'PreferDualStack' (two IP families on dual-stack configured clusters or a single IP family on single-stack clusters), or 'RequireDualStack' (two IP families on dual-stack configured clusters, otherwise fail). The ipFamilies and clusterIPs fields depend on the value of this field.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#ip_family_policy Service#ip_family_policy}
        '''
        result = self._values.get("ip_family_policy")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def load_balancer_class(self) -> typing.Optional[builtins.str]:
        '''The class of the load balancer implementation this Service belongs to.

        If specified, the value of this field must be a label-style identifier, with an optional prefix. This field can only be set when the Service type is ``LoadBalancer``. If not set, the default load balancer implementation is used. This field can only be set when creating or updating a Service to type ``LoadBalancer``. More info: https://kubernetes.io/docs/concepts/services-networking/service/#load-balancer-class

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#load_balancer_class Service#load_balancer_class}
        '''
        result = self._values.get("load_balancer_class")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def load_balancer_ip(self) -> typing.Optional[builtins.str]:
        '''Only applies to ``type = LoadBalancer``.

        LoadBalancer will get created with the IP specified in this field. This feature depends on whether the underlying cloud-provider supports specifying this field when a load balancer is created. This field will be ignored if the cloud-provider does not support the feature.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#load_balancer_ip Service#load_balancer_ip}
        '''
        result = self._values.get("load_balancer_ip")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def load_balancer_source_ranges(self) -> typing.Optional[typing.List[builtins.str]]:
        '''If specified and supported by the platform, this will restrict traffic through the cloud-provider load-balancer will be restricted to the specified client IPs.

        This field will be ignored if the cloud-provider does not support the feature. More info: http://kubernetes.io/docs/user-guide/services-firewalls

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#load_balancer_source_ranges Service#load_balancer_source_ranges}
        '''
        result = self._values.get("load_balancer_source_ranges")
        return typing.cast(typing.Optional[typing.List[builtins.str]], result)

    @builtins.property
    def port(
        self,
    ) -> typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.List["ServiceSpecPort"]]]:
        '''port block.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#port Service#port}
        '''
        result = self._values.get("port")
        return typing.cast(typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.List["ServiceSpecPort"]]], result)

    @builtins.property
    def publish_not_ready_addresses(
        self,
    ) -> typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]]:
        '''When set to true, indicates that DNS implementations must publish the ``notReadyAddresses`` of subsets for the Endpoints associated with the Service.

        The default value is ``false``. The primary use case for setting this field is to use a StatefulSet's Headless Service to propagate ``SRV`` records for its Pods without respect to their readiness for purpose of peer discovery.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#publish_not_ready_addresses Service#publish_not_ready_addresses}
        '''
        result = self._values.get("publish_not_ready_addresses")
        return typing.cast(typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]], result)

    @builtins.property
    def selector(self) -> typing.Optional[typing.Mapping[builtins.str, builtins.str]]:
        '''Route service traffic to pods with label keys and values matching this selector.

        Only applies to types ``ClusterIP``, ``NodePort``, and ``LoadBalancer``. More info: http://kubernetes.io/docs/user-guide/services#overview

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#selector Service#selector}
        '''
        result = self._values.get("selector")
        return typing.cast(typing.Optional[typing.Mapping[builtins.str, builtins.str]], result)

    @builtins.property
    def session_affinity(self) -> typing.Optional[builtins.str]:
        '''Used to maintain session affinity. Supports ``ClientIP`` and ``None``. Defaults to ``None``. More info: http://kubernetes.io/docs/user-guide/services#virtual-ips-and-service-proxies.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#session_affinity Service#session_affinity}
        '''
        result = self._values.get("session_affinity")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def session_affinity_config(
        self,
    ) -> typing.Optional["ServiceSpecSessionAffinityConfig"]:
        '''session_affinity_config block.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#session_affinity_config Service#session_affinity_config}
        '''
        result = self._values.get("session_affinity_config")
        return typing.cast(typing.Optional["ServiceSpecSessionAffinityConfig"], result)

    @builtins.property
    def type(self) -> typing.Optional[builtins.str]:
        '''Determines how the service is exposed.

        Defaults to ``ClusterIP``. Valid options are ``ExternalName``, ``ClusterIP``, ``NodePort``, and ``LoadBalancer``. ``ExternalName`` maps to the specified ``external_name``. More info: http://kubernetes.io/docs/user-guide/services#overview

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#type Service#type}
        '''
        result = self._values.get("type")
        return typing.cast(typing.Optional[builtins.str], result)

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceSpec(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


class ServiceSpecOutputReference(
    _cdktf_9a9027ec.ComplexObject,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceSpecOutputReference",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__ef17b13096e97c955edacd76e54ebada6a9004312f1d7ceca0c1625cb4fadee5)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute])

    @jsii.member(jsii_name="putPort")
    def put_port(
        self,
        value: typing.Union[_cdktf_9a9027ec.IResolvable, typing.Sequence[typing.Union["ServiceSpecPort", typing.Dict[builtins.str, typing.Any]]]],
    ) -> None:
        '''
        :param value: -
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__087915b9f41837b39fdb3d5efb9517a967fb502e6f5cd6c3bc6e72600907aeea)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        return typing.cast(None, jsii.invoke(self, "putPort", [value]))

    @jsii.member(jsii_name="putSessionAffinityConfig")
    def put_session_affinity_config(
        self,
        *,
        client_ip: typing.Optional[typing.Union["ServiceSpecSessionAffinityConfigClientIp", typing.Dict[builtins.str, typing.Any]]] = None,
    ) -> None:
        '''
        :param client_ip: client_ip block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#client_ip Service#client_ip}
        '''
        value = ServiceSpecSessionAffinityConfig(client_ip=client_ip)

        return typing.cast(None, jsii.invoke(self, "putSessionAffinityConfig", [value]))

    @jsii.member(jsii_name="resetAllocateLoadBalancerNodePorts")
    def reset_allocate_load_balancer_node_ports(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetAllocateLoadBalancerNodePorts", []))

    @jsii.member(jsii_name="resetClusterIp")
    def reset_cluster_ip(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetClusterIp", []))

    @jsii.member(jsii_name="resetClusterIps")
    def reset_cluster_ips(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetClusterIps", []))

    @jsii.member(jsii_name="resetExternalIps")
    def reset_external_ips(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetExternalIps", []))

    @jsii.member(jsii_name="resetExternalName")
    def reset_external_name(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetExternalName", []))

    @jsii.member(jsii_name="resetExternalTrafficPolicy")
    def reset_external_traffic_policy(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetExternalTrafficPolicy", []))

    @jsii.member(jsii_name="resetHealthCheckNodePort")
    def reset_health_check_node_port(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetHealthCheckNodePort", []))

    @jsii.member(jsii_name="resetInternalTrafficPolicy")
    def reset_internal_traffic_policy(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetInternalTrafficPolicy", []))

    @jsii.member(jsii_name="resetIpFamilies")
    def reset_ip_families(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetIpFamilies", []))

    @jsii.member(jsii_name="resetIpFamilyPolicy")
    def reset_ip_family_policy(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetIpFamilyPolicy", []))

    @jsii.member(jsii_name="resetLoadBalancerClass")
    def reset_load_balancer_class(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetLoadBalancerClass", []))

    @jsii.member(jsii_name="resetLoadBalancerIp")
    def reset_load_balancer_ip(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetLoadBalancerIp", []))

    @jsii.member(jsii_name="resetLoadBalancerSourceRanges")
    def reset_load_balancer_source_ranges(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetLoadBalancerSourceRanges", []))

    @jsii.member(jsii_name="resetPort")
    def reset_port(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetPort", []))

    @jsii.member(jsii_name="resetPublishNotReadyAddresses")
    def reset_publish_not_ready_addresses(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetPublishNotReadyAddresses", []))

    @jsii.member(jsii_name="resetSelector")
    def reset_selector(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetSelector", []))

    @jsii.member(jsii_name="resetSessionAffinity")
    def reset_session_affinity(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetSessionAffinity", []))

    @jsii.member(jsii_name="resetSessionAffinityConfig")
    def reset_session_affinity_config(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetSessionAffinityConfig", []))

    @jsii.member(jsii_name="resetType")
    def reset_type(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetType", []))

    @builtins.property
    @jsii.member(jsii_name="port")
    def port(self) -> "ServiceSpecPortList":
        return typing.cast("ServiceSpecPortList", jsii.get(self, "port"))

    @builtins.property
    @jsii.member(jsii_name="sessionAffinityConfig")
    def session_affinity_config(
        self,
    ) -> "ServiceSpecSessionAffinityConfigOutputReference":
        return typing.cast("ServiceSpecSessionAffinityConfigOutputReference", jsii.get(self, "sessionAffinityConfig"))

    @builtins.property
    @jsii.member(jsii_name="allocateLoadBalancerNodePortsInput")
    def allocate_load_balancer_node_ports_input(
        self,
    ) -> typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]]:
        return typing.cast(typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]], jsii.get(self, "allocateLoadBalancerNodePortsInput"))

    @builtins.property
    @jsii.member(jsii_name="clusterIpInput")
    def cluster_ip_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "clusterIpInput"))

    @builtins.property
    @jsii.member(jsii_name="clusterIpsInput")
    def cluster_ips_input(self) -> typing.Optional[typing.List[builtins.str]]:
        return typing.cast(typing.Optional[typing.List[builtins.str]], jsii.get(self, "clusterIpsInput"))

    @builtins.property
    @jsii.member(jsii_name="externalIpsInput")
    def external_ips_input(self) -> typing.Optional[typing.List[builtins.str]]:
        return typing.cast(typing.Optional[typing.List[builtins.str]], jsii.get(self, "externalIpsInput"))

    @builtins.property
    @jsii.member(jsii_name="externalNameInput")
    def external_name_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "externalNameInput"))

    @builtins.property
    @jsii.member(jsii_name="externalTrafficPolicyInput")
    def external_traffic_policy_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "externalTrafficPolicyInput"))

    @builtins.property
    @jsii.member(jsii_name="healthCheckNodePortInput")
    def health_check_node_port_input(self) -> typing.Optional[jsii.Number]:
        return typing.cast(typing.Optional[jsii.Number], jsii.get(self, "healthCheckNodePortInput"))

    @builtins.property
    @jsii.member(jsii_name="internalTrafficPolicyInput")
    def internal_traffic_policy_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "internalTrafficPolicyInput"))

    @builtins.property
    @jsii.member(jsii_name="ipFamiliesInput")
    def ip_families_input(self) -> typing.Optional[typing.List[builtins.str]]:
        return typing.cast(typing.Optional[typing.List[builtins.str]], jsii.get(self, "ipFamiliesInput"))

    @builtins.property
    @jsii.member(jsii_name="ipFamilyPolicyInput")
    def ip_family_policy_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "ipFamilyPolicyInput"))

    @builtins.property
    @jsii.member(jsii_name="loadBalancerClassInput")
    def load_balancer_class_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "loadBalancerClassInput"))

    @builtins.property
    @jsii.member(jsii_name="loadBalancerIpInput")
    def load_balancer_ip_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "loadBalancerIpInput"))

    @builtins.property
    @jsii.member(jsii_name="loadBalancerSourceRangesInput")
    def load_balancer_source_ranges_input(
        self,
    ) -> typing.Optional[typing.List[builtins.str]]:
        return typing.cast(typing.Optional[typing.List[builtins.str]], jsii.get(self, "loadBalancerSourceRangesInput"))

    @builtins.property
    @jsii.member(jsii_name="portInput")
    def port_input(
        self,
    ) -> typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.List["ServiceSpecPort"]]]:
        return typing.cast(typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.List["ServiceSpecPort"]]], jsii.get(self, "portInput"))

    @builtins.property
    @jsii.member(jsii_name="publishNotReadyAddressesInput")
    def publish_not_ready_addresses_input(
        self,
    ) -> typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]]:
        return typing.cast(typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]], jsii.get(self, "publishNotReadyAddressesInput"))

    @builtins.property
    @jsii.member(jsii_name="selectorInput")
    def selector_input(
        self,
    ) -> typing.Optional[typing.Mapping[builtins.str, builtins.str]]:
        return typing.cast(typing.Optional[typing.Mapping[builtins.str, builtins.str]], jsii.get(self, "selectorInput"))

    @builtins.property
    @jsii.member(jsii_name="sessionAffinityConfigInput")
    def session_affinity_config_input(
        self,
    ) -> typing.Optional["ServiceSpecSessionAffinityConfig"]:
        return typing.cast(typing.Optional["ServiceSpecSessionAffinityConfig"], jsii.get(self, "sessionAffinityConfigInput"))

    @builtins.property
    @jsii.member(jsii_name="sessionAffinityInput")
    def session_affinity_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "sessionAffinityInput"))

    @builtins.property
    @jsii.member(jsii_name="typeInput")
    def type_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "typeInput"))

    @builtins.property
    @jsii.member(jsii_name="allocateLoadBalancerNodePorts")
    def allocate_load_balancer_node_ports(
        self,
    ) -> typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]:
        return typing.cast(typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable], jsii.get(self, "allocateLoadBalancerNodePorts"))

    @allocate_load_balancer_node_ports.setter
    def allocate_load_balancer_node_ports(
        self,
        value: typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable],
    ) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__a4f73a743eff3c7b3a15e87754d12ece5c76127ae555560400ff4e3c14a66ca9)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "allocateLoadBalancerNodePorts", value)

    @builtins.property
    @jsii.member(jsii_name="clusterIp")
    def cluster_ip(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "clusterIp"))

    @cluster_ip.setter
    def cluster_ip(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__8e17ff5cde935e787a3ba9573e2bca2e5e2885b83006d57e0f1a00cdc1a16ae5)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "clusterIp", value)

    @builtins.property
    @jsii.member(jsii_name="clusterIps")
    def cluster_ips(self) -> typing.List[builtins.str]:
        return typing.cast(typing.List[builtins.str], jsii.get(self, "clusterIps"))

    @cluster_ips.setter
    def cluster_ips(self, value: typing.List[builtins.str]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__ffaec4b83256f07294f8a5ab40e8b4138fdc43ec53b864fbda4e846a978304aa)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "clusterIps", value)

    @builtins.property
    @jsii.member(jsii_name="externalIps")
    def external_ips(self) -> typing.List[builtins.str]:
        return typing.cast(typing.List[builtins.str], jsii.get(self, "externalIps"))

    @external_ips.setter
    def external_ips(self, value: typing.List[builtins.str]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__0a61a18194e02df61d3cfbcdc9925d1d031828b0c356cea4e9ce21be9c9ef823)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "externalIps", value)

    @builtins.property
    @jsii.member(jsii_name="externalName")
    def external_name(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "externalName"))

    @external_name.setter
    def external_name(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__c44d27c86046c4471bbdf5fc27b01e15ba8cf0bdbd355bcec7a03f747a9aa9b8)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "externalName", value)

    @builtins.property
    @jsii.member(jsii_name="externalTrafficPolicy")
    def external_traffic_policy(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "externalTrafficPolicy"))

    @external_traffic_policy.setter
    def external_traffic_policy(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__4f7cb1f153f104853fa730d0a89488a0626c253b244ead0e2254418cb1da67b6)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "externalTrafficPolicy", value)

    @builtins.property
    @jsii.member(jsii_name="healthCheckNodePort")
    def health_check_node_port(self) -> jsii.Number:
        return typing.cast(jsii.Number, jsii.get(self, "healthCheckNodePort"))

    @health_check_node_port.setter
    def health_check_node_port(self, value: jsii.Number) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__9548c75677907311e1bd63b49a11cfcc0d59768524f440493ad5ad51df1036f6)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "healthCheckNodePort", value)

    @builtins.property
    @jsii.member(jsii_name="internalTrafficPolicy")
    def internal_traffic_policy(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "internalTrafficPolicy"))

    @internal_traffic_policy.setter
    def internal_traffic_policy(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__a1800ac2e733251918179c22a92cc9a02c3bfa44fd676897eebc5794a459d007)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalTrafficPolicy", value)

    @builtins.property
    @jsii.member(jsii_name="ipFamilies")
    def ip_families(self) -> typing.List[builtins.str]:
        return typing.cast(typing.List[builtins.str], jsii.get(self, "ipFamilies"))

    @ip_families.setter
    def ip_families(self, value: typing.List[builtins.str]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__39b64d81b2a253bfce793b2d0bdfb92d3b269fa24d5b1ca38d18629ecb487e4d)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "ipFamilies", value)

    @builtins.property
    @jsii.member(jsii_name="ipFamilyPolicy")
    def ip_family_policy(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "ipFamilyPolicy"))

    @ip_family_policy.setter
    def ip_family_policy(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__206a2f05bf3d96ab2ea50a3e617571e32506b850c953141c74f91ed89d8c59f6)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "ipFamilyPolicy", value)

    @builtins.property
    @jsii.member(jsii_name="loadBalancerClass")
    def load_balancer_class(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "loadBalancerClass"))

    @load_balancer_class.setter
    def load_balancer_class(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__89f619f97a822d4b20d2090225946836f88fe5e4af1b616046e2e88ae16b1adb)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "loadBalancerClass", value)

    @builtins.property
    @jsii.member(jsii_name="loadBalancerIp")
    def load_balancer_ip(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "loadBalancerIp"))

    @load_balancer_ip.setter
    def load_balancer_ip(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__189dc7215e3d66cdc05cf1535dde9be5559dec91150fd22904bb300931e43da6)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "loadBalancerIp", value)

    @builtins.property
    @jsii.member(jsii_name="loadBalancerSourceRanges")
    def load_balancer_source_ranges(self) -> typing.List[builtins.str]:
        return typing.cast(typing.List[builtins.str], jsii.get(self, "loadBalancerSourceRanges"))

    @load_balancer_source_ranges.setter
    def load_balancer_source_ranges(self, value: typing.List[builtins.str]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__24ebbe4e5205fed5a3267f432d4e1dc7ea936f9da7bb33ef0ec086ed4a8d07b9)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "loadBalancerSourceRanges", value)

    @builtins.property
    @jsii.member(jsii_name="publishNotReadyAddresses")
    def publish_not_ready_addresses(
        self,
    ) -> typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]:
        return typing.cast(typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable], jsii.get(self, "publishNotReadyAddresses"))

    @publish_not_ready_addresses.setter
    def publish_not_ready_addresses(
        self,
        value: typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable],
    ) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__4eafc425ec1b1caef5d5f56c6fae791c7c42109a430a960cc2bca071e19ca224)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "publishNotReadyAddresses", value)

    @builtins.property
    @jsii.member(jsii_name="selector")
    def selector(self) -> typing.Mapping[builtins.str, builtins.str]:
        return typing.cast(typing.Mapping[builtins.str, builtins.str], jsii.get(self, "selector"))

    @selector.setter
    def selector(self, value: typing.Mapping[builtins.str, builtins.str]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__1cd7e0c4398f15e4b021f6f8f317496fa8714bdef02dc5074b7f7be34d004456)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "selector", value)

    @builtins.property
    @jsii.member(jsii_name="sessionAffinity")
    def session_affinity(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "sessionAffinity"))

    @session_affinity.setter
    def session_affinity(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__fa6243c723e96d5d2d31bfaa3e0bde1d40beac12141526c2ac84bf6dc9cbd3a6)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "sessionAffinity", value)

    @builtins.property
    @jsii.member(jsii_name="type")
    def type(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "type"))

    @type.setter
    def type(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__1be0a835868627f60c65b090ab48a5ac1c1176eb0b82898defd4cb3a94730e61)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "type", value)

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(self) -> typing.Optional[ServiceSpec]:
        return typing.cast(typing.Optional[ServiceSpec], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(self, value: typing.Optional[ServiceSpec]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__7e13e85b739ca2f109d527dd676625db847d4c3d2a04098bece381af25c0d533)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceSpecPort",
    jsii_struct_bases=[],
    name_mapping={
        "port": "port",
        "app_protocol": "appProtocol",
        "name": "name",
        "node_port": "nodePort",
        "protocol": "protocol",
        "target_port": "targetPort",
    },
)
class ServiceSpecPort:
    def __init__(
        self,
        *,
        port: jsii.Number,
        app_protocol: typing.Optional[builtins.str] = None,
        name: typing.Optional[builtins.str] = None,
        node_port: typing.Optional[jsii.Number] = None,
        protocol: typing.Optional[builtins.str] = None,
        target_port: typing.Optional[builtins.str] = None,
    ) -> None:
        '''
        :param port: The port that will be exposed by this service. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#port Service#port}
        :param app_protocol: The application protocol for this port. This field follows standard Kubernetes label syntax. Un-prefixed names are reserved for IANA standard service names (as per RFC-6335 and http://www.iana.org/assignments/service-names). Non-standard protocols should use prefixed names such as mycompany.com/my-custom-protocol. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#app_protocol Service#app_protocol}
        :param name: The name of this port within the service. All ports within the service must have unique names. Optional if only one ServicePort is defined on this service. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#name Service#name}
        :param node_port: The port on each node on which this service is exposed when ``type`` is ``NodePort`` or ``LoadBalancer``. Usually assigned by the system. If specified, it will be allocated to the service if unused or else creation of the service will fail. Default is to auto-allocate a port if the ``type`` of this service requires one. More info: http://kubernetes.io/docs/user-guide/services#type--nodeport Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#node_port Service#node_port}
        :param protocol: The IP protocol for this port. Supports ``TCP`` and ``UDP``. Default is ``TCP``. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#protocol Service#protocol}
        :param target_port: Number or name of the port to access on the pods targeted by the service. Number must be in the range 1 to 65535. This field is ignored for services with ``cluster_ip = "None"``. More info: http://kubernetes.io/docs/user-guide/services#defining-a-service Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#target_port Service#target_port}
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__dfa62c2f42e5704b799c2ab9b74fadaa889a38cc2bb43c1ebeb7fd24e4354481)
            check_type(argname="argument port", value=port, expected_type=type_hints["port"])
            check_type(argname="argument app_protocol", value=app_protocol, expected_type=type_hints["app_protocol"])
            check_type(argname="argument name", value=name, expected_type=type_hints["name"])
            check_type(argname="argument node_port", value=node_port, expected_type=type_hints["node_port"])
            check_type(argname="argument protocol", value=protocol, expected_type=type_hints["protocol"])
            check_type(argname="argument target_port", value=target_port, expected_type=type_hints["target_port"])
        self._values: typing.Dict[builtins.str, typing.Any] = {
            "port": port,
        }
        if app_protocol is not None:
            self._values["app_protocol"] = app_protocol
        if name is not None:
            self._values["name"] = name
        if node_port is not None:
            self._values["node_port"] = node_port
        if protocol is not None:
            self._values["protocol"] = protocol
        if target_port is not None:
            self._values["target_port"] = target_port

    @builtins.property
    def port(self) -> jsii.Number:
        '''The port that will be exposed by this service.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#port Service#port}
        '''
        result = self._values.get("port")
        assert result is not None, "Required property 'port' is missing"
        return typing.cast(jsii.Number, result)

    @builtins.property
    def app_protocol(self) -> typing.Optional[builtins.str]:
        '''The application protocol for this port.

        This field follows standard Kubernetes label syntax. Un-prefixed names are reserved for IANA standard service names (as per RFC-6335 and http://www.iana.org/assignments/service-names). Non-standard protocols should use prefixed names such as mycompany.com/my-custom-protocol.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#app_protocol Service#app_protocol}
        '''
        result = self._values.get("app_protocol")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def name(self) -> typing.Optional[builtins.str]:
        '''The name of this port within the service.

        All ports within the service must have unique names. Optional if only one ServicePort is defined on this service.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#name Service#name}
        '''
        result = self._values.get("name")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def node_port(self) -> typing.Optional[jsii.Number]:
        '''The port on each node on which this service is exposed when ``type`` is ``NodePort`` or ``LoadBalancer``.

        Usually assigned by the system. If specified, it will be allocated to the service if unused or else creation of the service will fail. Default is to auto-allocate a port if the ``type`` of this service requires one. More info: http://kubernetes.io/docs/user-guide/services#type--nodeport

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#node_port Service#node_port}
        '''
        result = self._values.get("node_port")
        return typing.cast(typing.Optional[jsii.Number], result)

    @builtins.property
    def protocol(self) -> typing.Optional[builtins.str]:
        '''The IP protocol for this port. Supports ``TCP`` and ``UDP``. Default is ``TCP``.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#protocol Service#protocol}
        '''
        result = self._values.get("protocol")
        return typing.cast(typing.Optional[builtins.str], result)

    @builtins.property
    def target_port(self) -> typing.Optional[builtins.str]:
        '''Number or name of the port to access on the pods targeted by the service.

        Number must be in the range 1 to 65535. This field is ignored for services with ``cluster_ip = "None"``. More info: http://kubernetes.io/docs/user-guide/services#defining-a-service

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#target_port Service#target_port}
        '''
        result = self._values.get("target_port")
        return typing.cast(typing.Optional[builtins.str], result)

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceSpecPort(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


class ServiceSpecPortList(
    _cdktf_9a9027ec.ComplexList,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceSpecPortList",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
        wraps_set: builtins.bool,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        :param wraps_set: whether the list is wrapping a set (will add tolist() to be able to access an item via an index).
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__14cbaf879a1ac09537a3f29f69184e9533cdc8949882d5c5d45a633159accf28)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
            check_type(argname="argument wraps_set", value=wraps_set, expected_type=type_hints["wraps_set"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute, wraps_set])

    @jsii.member(jsii_name="get")
    def get(self, index: jsii.Number) -> "ServiceSpecPortOutputReference":
        '''
        :param index: the index of the item to return.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__7d405849f5e0dce5a1057d8efc88127ba9bfcbfd3cf140327638932c73bda135)
            check_type(argname="argument index", value=index, expected_type=type_hints["index"])
        return typing.cast("ServiceSpecPortOutputReference", jsii.invoke(self, "get", [index]))

    @builtins.property
    @jsii.member(jsii_name="terraformAttribute")
    def _terraform_attribute(self) -> builtins.str:
        '''The attribute on the parent resource this class is referencing.'''
        return typing.cast(builtins.str, jsii.get(self, "terraformAttribute"))

    @_terraform_attribute.setter
    def _terraform_attribute(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__fc63ce87654afe099c1e7e30504e9dae9765df5510c8b98b94e5650a8c68e03b)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "terraformAttribute", value)

    @builtins.property
    @jsii.member(jsii_name="terraformResource")
    def _terraform_resource(self) -> _cdktf_9a9027ec.IInterpolatingParent:
        '''The parent resource.'''
        return typing.cast(_cdktf_9a9027ec.IInterpolatingParent, jsii.get(self, "terraformResource"))

    @_terraform_resource.setter
    def _terraform_resource(self, value: _cdktf_9a9027ec.IInterpolatingParent) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__61a17ba446ef217d9b7ad56c782c00610b462636522cd72866b542d783ebf3ba)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "terraformResource", value)

    @builtins.property
    @jsii.member(jsii_name="wrapsSet")
    def _wraps_set(self) -> builtins.bool:
        '''whether the list is wrapping a set (will add tolist() to be able to access an item via an index).'''
        return typing.cast(builtins.bool, jsii.get(self, "wrapsSet"))

    @_wraps_set.setter
    def _wraps_set(self, value: builtins.bool) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__86b83c05a809d14c65ac7e9aea6958228d0a50ae00f45d0236637a46e8234c3e)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "wrapsSet", value)

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(
        self,
    ) -> typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.List[ServiceSpecPort]]]:
        return typing.cast(typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.List[ServiceSpecPort]]], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(
        self,
        value: typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.List[ServiceSpecPort]]],
    ) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__36d4209fb9ca6b2eaf5ef1ef73238f18936f32a28e2c7f7ae69c5fdd954342e2)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


class ServiceSpecPortOutputReference(
    _cdktf_9a9027ec.ComplexObject,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceSpecPortOutputReference",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
        complex_object_index: jsii.Number,
        complex_object_is_from_set: builtins.bool,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        :param complex_object_index: the index of this item in the list.
        :param complex_object_is_from_set: whether the list is wrapping a set (will add tolist() to be able to access an item via an index).
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__4525fd4094001093f1f89c9b67770297099d96afda372e1d9b00e4f0ec221079)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
            check_type(argname="argument complex_object_index", value=complex_object_index, expected_type=type_hints["complex_object_index"])
            check_type(argname="argument complex_object_is_from_set", value=complex_object_is_from_set, expected_type=type_hints["complex_object_is_from_set"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute, complex_object_index, complex_object_is_from_set])

    @jsii.member(jsii_name="resetAppProtocol")
    def reset_app_protocol(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetAppProtocol", []))

    @jsii.member(jsii_name="resetName")
    def reset_name(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetName", []))

    @jsii.member(jsii_name="resetNodePort")
    def reset_node_port(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetNodePort", []))

    @jsii.member(jsii_name="resetProtocol")
    def reset_protocol(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetProtocol", []))

    @jsii.member(jsii_name="resetTargetPort")
    def reset_target_port(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetTargetPort", []))

    @builtins.property
    @jsii.member(jsii_name="appProtocolInput")
    def app_protocol_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "appProtocolInput"))

    @builtins.property
    @jsii.member(jsii_name="nameInput")
    def name_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "nameInput"))

    @builtins.property
    @jsii.member(jsii_name="nodePortInput")
    def node_port_input(self) -> typing.Optional[jsii.Number]:
        return typing.cast(typing.Optional[jsii.Number], jsii.get(self, "nodePortInput"))

    @builtins.property
    @jsii.member(jsii_name="portInput")
    def port_input(self) -> typing.Optional[jsii.Number]:
        return typing.cast(typing.Optional[jsii.Number], jsii.get(self, "portInput"))

    @builtins.property
    @jsii.member(jsii_name="protocolInput")
    def protocol_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "protocolInput"))

    @builtins.property
    @jsii.member(jsii_name="targetPortInput")
    def target_port_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "targetPortInput"))

    @builtins.property
    @jsii.member(jsii_name="appProtocol")
    def app_protocol(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "appProtocol"))

    @app_protocol.setter
    def app_protocol(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__83d4c9a221bf213c6ec6a3c74942c1f05b344ba1c8025dd479cc2b766013eabb)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "appProtocol", value)

    @builtins.property
    @jsii.member(jsii_name="name")
    def name(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "name"))

    @name.setter
    def name(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__2ecc771ffde6b49da755506dca496777e78528f0bb6b1d86d2d1db9fab46a294)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "name", value)

    @builtins.property
    @jsii.member(jsii_name="nodePort")
    def node_port(self) -> jsii.Number:
        return typing.cast(jsii.Number, jsii.get(self, "nodePort"))

    @node_port.setter
    def node_port(self, value: jsii.Number) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__7a164de8289e6184e3ca54bfdda067a9f232b295a7178a7a4d266f09f08f6e06)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "nodePort", value)

    @builtins.property
    @jsii.member(jsii_name="port")
    def port(self) -> jsii.Number:
        return typing.cast(jsii.Number, jsii.get(self, "port"))

    @port.setter
    def port(self, value: jsii.Number) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__967caac236bbae765fea0b47ee18f7449deaa64e224c77daee73a23050146bf7)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "port", value)

    @builtins.property
    @jsii.member(jsii_name="protocol")
    def protocol(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "protocol"))

    @protocol.setter
    def protocol(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__9175f51d176ec09bb6c7ffd0c645b36d9f2f08e906739fa66e4bbdcb313759cc)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "protocol", value)

    @builtins.property
    @jsii.member(jsii_name="targetPort")
    def target_port(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "targetPort"))

    @target_port.setter
    def target_port(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__aa467c462d554f6377c9315b470e518036c31fedd0384f904c24c2ef82bee15c)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "targetPort", value)

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(
        self,
    ) -> typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, ServiceSpecPort]]:
        return typing.cast(typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, ServiceSpecPort]], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(
        self,
        value: typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, ServiceSpecPort]],
    ) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__7ef6cd646d56464bfa46a6c821b3c761b9f762f737ff996bc199a42639477048)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceSpecSessionAffinityConfig",
    jsii_struct_bases=[],
    name_mapping={"client_ip": "clientIp"},
)
class ServiceSpecSessionAffinityConfig:
    def __init__(
        self,
        *,
        client_ip: typing.Optional[typing.Union["ServiceSpecSessionAffinityConfigClientIp", typing.Dict[builtins.str, typing.Any]]] = None,
    ) -> None:
        '''
        :param client_ip: client_ip block. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#client_ip Service#client_ip}
        '''
        if isinstance(client_ip, dict):
            client_ip = ServiceSpecSessionAffinityConfigClientIp(**client_ip)
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__56cbf679d417c53baf9f20df11ca36d141bc94c346adef4803ae16d614a64cb6)
            check_type(argname="argument client_ip", value=client_ip, expected_type=type_hints["client_ip"])
        self._values: typing.Dict[builtins.str, typing.Any] = {}
        if client_ip is not None:
            self._values["client_ip"] = client_ip

    @builtins.property
    def client_ip(self) -> typing.Optional["ServiceSpecSessionAffinityConfigClientIp"]:
        '''client_ip block.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#client_ip Service#client_ip}
        '''
        result = self._values.get("client_ip")
        return typing.cast(typing.Optional["ServiceSpecSessionAffinityConfigClientIp"], result)

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceSpecSessionAffinityConfig(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceSpecSessionAffinityConfigClientIp",
    jsii_struct_bases=[],
    name_mapping={"timeout_seconds": "timeoutSeconds"},
)
class ServiceSpecSessionAffinityConfigClientIp:
    def __init__(self, *, timeout_seconds: typing.Optional[jsii.Number] = None) -> None:
        '''
        :param timeout_seconds: Specifies the seconds of ``ClientIP`` type session sticky time. The value must be > 0 and <= 86400(for 1 day) if ``ServiceAffinity`` == ``ClientIP``. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#timeout_seconds Service#timeout_seconds}
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__ca5cb91d01776dd3876de5dd03af431c67c1cd2f99c884b7aa21d2af739d6a48)
            check_type(argname="argument timeout_seconds", value=timeout_seconds, expected_type=type_hints["timeout_seconds"])
        self._values: typing.Dict[builtins.str, typing.Any] = {}
        if timeout_seconds is not None:
            self._values["timeout_seconds"] = timeout_seconds

    @builtins.property
    def timeout_seconds(self) -> typing.Optional[jsii.Number]:
        '''Specifies the seconds of ``ClientIP`` type session sticky time.

        The value must be > 0 and <= 86400(for 1 day) if ``ServiceAffinity`` == ``ClientIP``.

        Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#timeout_seconds Service#timeout_seconds}
        '''
        result = self._values.get("timeout_seconds")
        return typing.cast(typing.Optional[jsii.Number], result)

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceSpecSessionAffinityConfigClientIp(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


class ServiceSpecSessionAffinityConfigClientIpOutputReference(
    _cdktf_9a9027ec.ComplexObject,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceSpecSessionAffinityConfigClientIpOutputReference",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__29bb1b9a02cd3cd872f3ba26979b1f9ccc1da0ca8d4acd5b1431499de2701436)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute])

    @jsii.member(jsii_name="resetTimeoutSeconds")
    def reset_timeout_seconds(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetTimeoutSeconds", []))

    @builtins.property
    @jsii.member(jsii_name="timeoutSecondsInput")
    def timeout_seconds_input(self) -> typing.Optional[jsii.Number]:
        return typing.cast(typing.Optional[jsii.Number], jsii.get(self, "timeoutSecondsInput"))

    @builtins.property
    @jsii.member(jsii_name="timeoutSeconds")
    def timeout_seconds(self) -> jsii.Number:
        return typing.cast(jsii.Number, jsii.get(self, "timeoutSeconds"))

    @timeout_seconds.setter
    def timeout_seconds(self, value: jsii.Number) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__86c6de3b9dd402fe6fd3a40ce4b950e7e60e74efd49ee69c8e96074a9c895722)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "timeoutSeconds", value)

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(
        self,
    ) -> typing.Optional[ServiceSpecSessionAffinityConfigClientIp]:
        return typing.cast(typing.Optional[ServiceSpecSessionAffinityConfigClientIp], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(
        self,
        value: typing.Optional[ServiceSpecSessionAffinityConfigClientIp],
    ) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__e546be226f23a44979cee419329497fa4034a797ddd37e9c638a67fa385b8112)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


class ServiceSpecSessionAffinityConfigOutputReference(
    _cdktf_9a9027ec.ComplexObject,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceSpecSessionAffinityConfigOutputReference",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__373241708a7d5153ec2a6ef06a2d01dad4697869f2b13ac35e70b34716a078f5)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute])

    @jsii.member(jsii_name="putClientIp")
    def put_client_ip(
        self,
        *,
        timeout_seconds: typing.Optional[jsii.Number] = None,
    ) -> None:
        '''
        :param timeout_seconds: Specifies the seconds of ``ClientIP`` type session sticky time. The value must be > 0 and <= 86400(for 1 day) if ``ServiceAffinity`` == ``ClientIP``. Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#timeout_seconds Service#timeout_seconds}
        '''
        value = ServiceSpecSessionAffinityConfigClientIp(
            timeout_seconds=timeout_seconds
        )

        return typing.cast(None, jsii.invoke(self, "putClientIp", [value]))

    @jsii.member(jsii_name="resetClientIp")
    def reset_client_ip(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetClientIp", []))

    @builtins.property
    @jsii.member(jsii_name="clientIp")
    def client_ip(self) -> ServiceSpecSessionAffinityConfigClientIpOutputReference:
        return typing.cast(ServiceSpecSessionAffinityConfigClientIpOutputReference, jsii.get(self, "clientIp"))

    @builtins.property
    @jsii.member(jsii_name="clientIpInput")
    def client_ip_input(
        self,
    ) -> typing.Optional[ServiceSpecSessionAffinityConfigClientIp]:
        return typing.cast(typing.Optional[ServiceSpecSessionAffinityConfigClientIp], jsii.get(self, "clientIpInput"))

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(self) -> typing.Optional[ServiceSpecSessionAffinityConfig]:
        return typing.cast(typing.Optional[ServiceSpecSessionAffinityConfig], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(
        self,
        value: typing.Optional[ServiceSpecSessionAffinityConfig],
    ) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__3557db88b732f7218c9fe763db2f40388468f1e0b0b6f4c0f7d9497e615d4855)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceStatus",
    jsii_struct_bases=[],
    name_mapping={},
)
class ServiceStatus:
    def __init__(self) -> None:
        self._values: typing.Dict[builtins.str, typing.Any] = {}

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceStatus(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


class ServiceStatusList(
    _cdktf_9a9027ec.ComplexList,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceStatusList",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
        wraps_set: builtins.bool,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        :param wraps_set: whether the list is wrapping a set (will add tolist() to be able to access an item via an index).
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__65e34c91f40dd12b785184ff640f81256d717eac8132a0f7070c9a9a8ffdf907)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
            check_type(argname="argument wraps_set", value=wraps_set, expected_type=type_hints["wraps_set"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute, wraps_set])

    @jsii.member(jsii_name="get")
    def get(self, index: jsii.Number) -> "ServiceStatusOutputReference":
        '''
        :param index: the index of the item to return.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__4ccd056e228d0359e708559fa6ac7b39f84a318233be854b8fb69ec344348ace)
            check_type(argname="argument index", value=index, expected_type=type_hints["index"])
        return typing.cast("ServiceStatusOutputReference", jsii.invoke(self, "get", [index]))

    @builtins.property
    @jsii.member(jsii_name="terraformAttribute")
    def _terraform_attribute(self) -> builtins.str:
        '''The attribute on the parent resource this class is referencing.'''
        return typing.cast(builtins.str, jsii.get(self, "terraformAttribute"))

    @_terraform_attribute.setter
    def _terraform_attribute(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__80e416b8b7ee7aece9b574322a9e0e2bbd7b4d97baedb617a0cb8ac48427844f)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "terraformAttribute", value)

    @builtins.property
    @jsii.member(jsii_name="terraformResource")
    def _terraform_resource(self) -> _cdktf_9a9027ec.IInterpolatingParent:
        '''The parent resource.'''
        return typing.cast(_cdktf_9a9027ec.IInterpolatingParent, jsii.get(self, "terraformResource"))

    @_terraform_resource.setter
    def _terraform_resource(self, value: _cdktf_9a9027ec.IInterpolatingParent) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__267e98f31c7d4340b6df1eac29f0077da25a432f0e255aa978bdab0aa4bc5d76)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "terraformResource", value)

    @builtins.property
    @jsii.member(jsii_name="wrapsSet")
    def _wraps_set(self) -> builtins.bool:
        '''whether the list is wrapping a set (will add tolist() to be able to access an item via an index).'''
        return typing.cast(builtins.bool, jsii.get(self, "wrapsSet"))

    @_wraps_set.setter
    def _wraps_set(self, value: builtins.bool) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__861247469de653396d1694235b3883b5dbbbf2e9c73e389aeb8dd68c538b6189)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "wrapsSet", value)


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceStatusLoadBalancer",
    jsii_struct_bases=[],
    name_mapping={},
)
class ServiceStatusLoadBalancer:
    def __init__(self) -> None:
        self._values: typing.Dict[builtins.str, typing.Any] = {}

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceStatusLoadBalancer(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceStatusLoadBalancerIngress",
    jsii_struct_bases=[],
    name_mapping={},
)
class ServiceStatusLoadBalancerIngress:
    def __init__(self) -> None:
        self._values: typing.Dict[builtins.str, typing.Any] = {}

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceStatusLoadBalancerIngress(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


class ServiceStatusLoadBalancerIngressList(
    _cdktf_9a9027ec.ComplexList,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceStatusLoadBalancerIngressList",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
        wraps_set: builtins.bool,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        :param wraps_set: whether the list is wrapping a set (will add tolist() to be able to access an item via an index).
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__e60cc630f1394ad52d1989c2ec3c691e96ba49d38a654d0fa91bfb33fda84ccf)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
            check_type(argname="argument wraps_set", value=wraps_set, expected_type=type_hints["wraps_set"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute, wraps_set])

    @jsii.member(jsii_name="get")
    def get(
        self,
        index: jsii.Number,
    ) -> "ServiceStatusLoadBalancerIngressOutputReference":
        '''
        :param index: the index of the item to return.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__28f6eb2a4afa45a81f2b82b4cdb4e6945188d798813540781f95149214b3e966)
            check_type(argname="argument index", value=index, expected_type=type_hints["index"])
        return typing.cast("ServiceStatusLoadBalancerIngressOutputReference", jsii.invoke(self, "get", [index]))

    @builtins.property
    @jsii.member(jsii_name="terraformAttribute")
    def _terraform_attribute(self) -> builtins.str:
        '''The attribute on the parent resource this class is referencing.'''
        return typing.cast(builtins.str, jsii.get(self, "terraformAttribute"))

    @_terraform_attribute.setter
    def _terraform_attribute(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__41778733a629be32f76298ec99ef749e860f664c86c762b7bf8da2fd5917f20f)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "terraformAttribute", value)

    @builtins.property
    @jsii.member(jsii_name="terraformResource")
    def _terraform_resource(self) -> _cdktf_9a9027ec.IInterpolatingParent:
        '''The parent resource.'''
        return typing.cast(_cdktf_9a9027ec.IInterpolatingParent, jsii.get(self, "terraformResource"))

    @_terraform_resource.setter
    def _terraform_resource(self, value: _cdktf_9a9027ec.IInterpolatingParent) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__e42be57dc3bc0c449bb176ce88c3a06f22dae5311bdc11b8c9e97cba0512e841)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "terraformResource", value)

    @builtins.property
    @jsii.member(jsii_name="wrapsSet")
    def _wraps_set(self) -> builtins.bool:
        '''whether the list is wrapping a set (will add tolist() to be able to access an item via an index).'''
        return typing.cast(builtins.bool, jsii.get(self, "wrapsSet"))

    @_wraps_set.setter
    def _wraps_set(self, value: builtins.bool) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__46b68cf4cdde48ab0042d240a5f14af5379607afd5453612a83f09e9396bc8f4)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "wrapsSet", value)


class ServiceStatusLoadBalancerIngressOutputReference(
    _cdktf_9a9027ec.ComplexObject,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceStatusLoadBalancerIngressOutputReference",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
        complex_object_index: jsii.Number,
        complex_object_is_from_set: builtins.bool,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        :param complex_object_index: the index of this item in the list.
        :param complex_object_is_from_set: whether the list is wrapping a set (will add tolist() to be able to access an item via an index).
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__fcacbe5baead16373d11976b649428a2c57749ecc00b1a4f6c3f674eb67128f9)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
            check_type(argname="argument complex_object_index", value=complex_object_index, expected_type=type_hints["complex_object_index"])
            check_type(argname="argument complex_object_is_from_set", value=complex_object_is_from_set, expected_type=type_hints["complex_object_is_from_set"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute, complex_object_index, complex_object_is_from_set])

    @builtins.property
    @jsii.member(jsii_name="hostname")
    def hostname(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "hostname"))

    @builtins.property
    @jsii.member(jsii_name="ip")
    def ip(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "ip"))

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(self) -> typing.Optional[ServiceStatusLoadBalancerIngress]:
        return typing.cast(typing.Optional[ServiceStatusLoadBalancerIngress], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(
        self,
        value: typing.Optional[ServiceStatusLoadBalancerIngress],
    ) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__ca28450224ec38f89194f2b121396445b57df7d177430c70f1a5120e594f9287)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


class ServiceStatusLoadBalancerList(
    _cdktf_9a9027ec.ComplexList,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceStatusLoadBalancerList",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
        wraps_set: builtins.bool,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        :param wraps_set: whether the list is wrapping a set (will add tolist() to be able to access an item via an index).
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__80bc58cca9608a2c356efbcc9571ddbdc2bd15980ec1d3e1dd070640117d14fa)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
            check_type(argname="argument wraps_set", value=wraps_set, expected_type=type_hints["wraps_set"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute, wraps_set])

    @jsii.member(jsii_name="get")
    def get(self, index: jsii.Number) -> "ServiceStatusLoadBalancerOutputReference":
        '''
        :param index: the index of the item to return.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__e90dd495578b6e2005d83f9bfc1219b5c1bf3fce6afd31eb88ddb291e4b29cba)
            check_type(argname="argument index", value=index, expected_type=type_hints["index"])
        return typing.cast("ServiceStatusLoadBalancerOutputReference", jsii.invoke(self, "get", [index]))

    @builtins.property
    @jsii.member(jsii_name="terraformAttribute")
    def _terraform_attribute(self) -> builtins.str:
        '''The attribute on the parent resource this class is referencing.'''
        return typing.cast(builtins.str, jsii.get(self, "terraformAttribute"))

    @_terraform_attribute.setter
    def _terraform_attribute(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__8d705a995ef12cf93668e9ffaef340696827d3ce452fe0ec245ce29f968676e5)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "terraformAttribute", value)

    @builtins.property
    @jsii.member(jsii_name="terraformResource")
    def _terraform_resource(self) -> _cdktf_9a9027ec.IInterpolatingParent:
        '''The parent resource.'''
        return typing.cast(_cdktf_9a9027ec.IInterpolatingParent, jsii.get(self, "terraformResource"))

    @_terraform_resource.setter
    def _terraform_resource(self, value: _cdktf_9a9027ec.IInterpolatingParent) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__e9d8f03b0c35d15e14e9c14d7bc9b14bbd62fd5398caa1ea2f677cfcaeb43316)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "terraformResource", value)

    @builtins.property
    @jsii.member(jsii_name="wrapsSet")
    def _wraps_set(self) -> builtins.bool:
        '''whether the list is wrapping a set (will add tolist() to be able to access an item via an index).'''
        return typing.cast(builtins.bool, jsii.get(self, "wrapsSet"))

    @_wraps_set.setter
    def _wraps_set(self, value: builtins.bool) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__1ea0f3ecda6159d148b9e50604e0fc959d2a9718827e85aa4d4448d33996e974)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "wrapsSet", value)


class ServiceStatusLoadBalancerOutputReference(
    _cdktf_9a9027ec.ComplexObject,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceStatusLoadBalancerOutputReference",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
        complex_object_index: jsii.Number,
        complex_object_is_from_set: builtins.bool,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        :param complex_object_index: the index of this item in the list.
        :param complex_object_is_from_set: whether the list is wrapping a set (will add tolist() to be able to access an item via an index).
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__496254de608e31c6b8c1198f2136fb23495f72c4bf00dde96e80e03aff490d67)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
            check_type(argname="argument complex_object_index", value=complex_object_index, expected_type=type_hints["complex_object_index"])
            check_type(argname="argument complex_object_is_from_set", value=complex_object_is_from_set, expected_type=type_hints["complex_object_is_from_set"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute, complex_object_index, complex_object_is_from_set])

    @builtins.property
    @jsii.member(jsii_name="ingress")
    def ingress(self) -> ServiceStatusLoadBalancerIngressList:
        return typing.cast(ServiceStatusLoadBalancerIngressList, jsii.get(self, "ingress"))

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(self) -> typing.Optional[ServiceStatusLoadBalancer]:
        return typing.cast(typing.Optional[ServiceStatusLoadBalancer], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(self, value: typing.Optional[ServiceStatusLoadBalancer]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__7ad9a2c887f2889e37ab43bd14fe8bb0b64811d56ed7549aaacbbed42f1a40e1)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


class ServiceStatusOutputReference(
    _cdktf_9a9027ec.ComplexObject,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceStatusOutputReference",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
        complex_object_index: jsii.Number,
        complex_object_is_from_set: builtins.bool,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        :param complex_object_index: the index of this item in the list.
        :param complex_object_is_from_set: whether the list is wrapping a set (will add tolist() to be able to access an item via an index).
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__e83b04ed2f4a3f18d905edcc9cbcc2f37d11afecffde28c00a8aca3517f37562)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
            check_type(argname="argument complex_object_index", value=complex_object_index, expected_type=type_hints["complex_object_index"])
            check_type(argname="argument complex_object_is_from_set", value=complex_object_is_from_set, expected_type=type_hints["complex_object_is_from_set"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute, complex_object_index, complex_object_is_from_set])

    @builtins.property
    @jsii.member(jsii_name="loadBalancer")
    def load_balancer(self) -> ServiceStatusLoadBalancerList:
        return typing.cast(ServiceStatusLoadBalancerList, jsii.get(self, "loadBalancer"))

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(self) -> typing.Optional[ServiceStatus]:
        return typing.cast(typing.Optional[ServiceStatus], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(self, value: typing.Optional[ServiceStatus]) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__ff5cc3c03637a9bdd641f4ed9621ed5a7c7ca504db30f17fe024973b44b6f70a)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


@jsii.data_type(
    jsii_type="kubernetes.service.ServiceTimeouts",
    jsii_struct_bases=[],
    name_mapping={"create": "create"},
)
class ServiceTimeouts:
    def __init__(self, *, create: typing.Optional[builtins.str] = None) -> None:
        '''
        :param create: Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#create Service#create}.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__a00dd7b62182a7deaa5344460346ae116d8c1e2d34aa66e4afb3722f488930d4)
            check_type(argname="argument create", value=create, expected_type=type_hints["create"])
        self._values: typing.Dict[builtins.str, typing.Any] = {}
        if create is not None:
            self._values["create"] = create

    @builtins.property
    def create(self) -> typing.Optional[builtins.str]:
        '''Docs at Terraform Registry: {@link https://registry.terraform.io/providers/hashicorp/kubernetes/2.22.0/docs/resources/service#create Service#create}.'''
        result = self._values.get("create")
        return typing.cast(typing.Optional[builtins.str], result)

    def __eq__(self, rhs: typing.Any) -> builtins.bool:
        return isinstance(rhs, self.__class__) and rhs._values == self._values

    def __ne__(self, rhs: typing.Any) -> builtins.bool:
        return not (rhs == self)

    def __repr__(self) -> str:
        return "ServiceTimeouts(%s)" % ", ".join(
            k + "=" + repr(v) for k, v in self._values.items()
        )


class ServiceTimeoutsOutputReference(
    _cdktf_9a9027ec.ComplexObject,
    metaclass=jsii.JSIIMeta,
    jsii_type="kubernetes.service.ServiceTimeoutsOutputReference",
):
    def __init__(
        self,
        terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
        terraform_attribute: builtins.str,
    ) -> None:
        '''
        :param terraform_resource: The parent resource.
        :param terraform_attribute: The attribute on the parent resource this class is referencing.
        '''
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__c88e3c75267dbaba86985d4faf30e357e15bdcbc681a372dc79afbbe6b4ab07d)
            check_type(argname="argument terraform_resource", value=terraform_resource, expected_type=type_hints["terraform_resource"])
            check_type(argname="argument terraform_attribute", value=terraform_attribute, expected_type=type_hints["terraform_attribute"])
        jsii.create(self.__class__, self, [terraform_resource, terraform_attribute])

    @jsii.member(jsii_name="resetCreate")
    def reset_create(self) -> None:
        return typing.cast(None, jsii.invoke(self, "resetCreate", []))

    @builtins.property
    @jsii.member(jsii_name="createInput")
    def create_input(self) -> typing.Optional[builtins.str]:
        return typing.cast(typing.Optional[builtins.str], jsii.get(self, "createInput"))

    @builtins.property
    @jsii.member(jsii_name="create")
    def create(self) -> builtins.str:
        return typing.cast(builtins.str, jsii.get(self, "create"))

    @create.setter
    def create(self, value: builtins.str) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__e57a48d12a6d923b6866e9ed2457e0814d69d8c577d3823f61c5f1df0d34b512)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "create", value)

    @builtins.property
    @jsii.member(jsii_name="internalValue")
    def internal_value(
        self,
    ) -> typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, ServiceTimeouts]]:
        return typing.cast(typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, ServiceTimeouts]], jsii.get(self, "internalValue"))

    @internal_value.setter
    def internal_value(
        self,
        value: typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, ServiceTimeouts]],
    ) -> None:
        if __debug__:
            type_hints = typing.get_type_hints(_typecheckingstub__989498eadfd8b1d4f89804fa3d3c0636d94443b0f4b2b284eeee21b4e6fee1d5)
            check_type(argname="argument value", value=value, expected_type=type_hints["value"])
        jsii.set(self, "internalValue", value)


__all__ = [
    "Service",
    "ServiceConfig",
    "ServiceMetadata",
    "ServiceMetadataOutputReference",
    "ServiceSpec",
    "ServiceSpecOutputReference",
    "ServiceSpecPort",
    "ServiceSpecPortList",
    "ServiceSpecPortOutputReference",
    "ServiceSpecSessionAffinityConfig",
    "ServiceSpecSessionAffinityConfigClientIp",
    "ServiceSpecSessionAffinityConfigClientIpOutputReference",
    "ServiceSpecSessionAffinityConfigOutputReference",
    "ServiceStatus",
    "ServiceStatusList",
    "ServiceStatusLoadBalancer",
    "ServiceStatusLoadBalancerIngress",
    "ServiceStatusLoadBalancerIngressList",
    "ServiceStatusLoadBalancerIngressOutputReference",
    "ServiceStatusLoadBalancerList",
    "ServiceStatusLoadBalancerOutputReference",
    "ServiceStatusOutputReference",
    "ServiceTimeouts",
    "ServiceTimeoutsOutputReference",
]

publication.publish()

def _typecheckingstub__9379740d5a41c67727672ceaa889c18038f8f9e9ea25dc72641d1e42e916ce26(
    scope: _constructs_77d1e7e8.Construct,
    id_: builtins.str,
    *,
    metadata: typing.Union[ServiceMetadata, typing.Dict[builtins.str, typing.Any]],
    spec: typing.Union[ServiceSpec, typing.Dict[builtins.str, typing.Any]],
    id: typing.Optional[builtins.str] = None,
    timeouts: typing.Optional[typing.Union[ServiceTimeouts, typing.Dict[builtins.str, typing.Any]]] = None,
    wait_for_load_balancer: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
    connection: typing.Optional[typing.Union[typing.Union[_cdktf_9a9027ec.SSHProvisionerConnection, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.WinrmProvisionerConnection, typing.Dict[builtins.str, typing.Any]]]] = None,
    count: typing.Optional[typing.Union[jsii.Number, _cdktf_9a9027ec.TerraformCount]] = None,
    depends_on: typing.Optional[typing.Sequence[_cdktf_9a9027ec.ITerraformDependable]] = None,
    for_each: typing.Optional[_cdktf_9a9027ec.ITerraformIterator] = None,
    lifecycle: typing.Optional[typing.Union[_cdktf_9a9027ec.TerraformResourceLifecycle, typing.Dict[builtins.str, typing.Any]]] = None,
    provider: typing.Optional[_cdktf_9a9027ec.TerraformProvider] = None,
    provisioners: typing.Optional[typing.Sequence[typing.Union[typing.Union[_cdktf_9a9027ec.FileProvisioner, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.LocalExecProvisioner, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.RemoteExecProvisioner, typing.Dict[builtins.str, typing.Any]]]]] = None,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__e7022b3079da14277606eaf8ed8a1a3d324bc22bc24e8c5416be6eaf49bd5dd4(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__80cb12f70e7be78328ea4c478ae9dc647443f48b01a34bf654ed2d9a59d00c14(
    value: typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__5705bb667ac158d1879c1a562c6e3206deb51c5f25f09b6f0d1223e19deef331(
    *,
    connection: typing.Optional[typing.Union[typing.Union[_cdktf_9a9027ec.SSHProvisionerConnection, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.WinrmProvisionerConnection, typing.Dict[builtins.str, typing.Any]]]] = None,
    count: typing.Optional[typing.Union[jsii.Number, _cdktf_9a9027ec.TerraformCount]] = None,
    depends_on: typing.Optional[typing.Sequence[_cdktf_9a9027ec.ITerraformDependable]] = None,
    for_each: typing.Optional[_cdktf_9a9027ec.ITerraformIterator] = None,
    lifecycle: typing.Optional[typing.Union[_cdktf_9a9027ec.TerraformResourceLifecycle, typing.Dict[builtins.str, typing.Any]]] = None,
    provider: typing.Optional[_cdktf_9a9027ec.TerraformProvider] = None,
    provisioners: typing.Optional[typing.Sequence[typing.Union[typing.Union[_cdktf_9a9027ec.FileProvisioner, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.LocalExecProvisioner, typing.Dict[builtins.str, typing.Any]], typing.Union[_cdktf_9a9027ec.RemoteExecProvisioner, typing.Dict[builtins.str, typing.Any]]]]] = None,
    metadata: typing.Union[ServiceMetadata, typing.Dict[builtins.str, typing.Any]],
    spec: typing.Union[ServiceSpec, typing.Dict[builtins.str, typing.Any]],
    id: typing.Optional[builtins.str] = None,
    timeouts: typing.Optional[typing.Union[ServiceTimeouts, typing.Dict[builtins.str, typing.Any]]] = None,
    wait_for_load_balancer: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__f83ac22dc025dff9cab6cf84ee3b09e78e5651e681b883d00a283cfb04e13a91(
    *,
    annotations: typing.Optional[typing.Mapping[builtins.str, builtins.str]] = None,
    generate_name: typing.Optional[builtins.str] = None,
    labels: typing.Optional[typing.Mapping[builtins.str, builtins.str]] = None,
    name: typing.Optional[builtins.str] = None,
    namespace: typing.Optional[builtins.str] = None,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__2272ee7a402d6118de95eb47fb9d04a3ed21a1dbdfd036db7acafe4bd4e4c0b9(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__10d98e551c5a15e59e1f1b47b622816536e176c8724d89d3fbf62c34ab64e283(
    value: typing.Mapping[builtins.str, builtins.str],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__9c4ea2d9c52da51e832a122907ba3bb96ce3b3fea4419b44d80b3f8bbaf42d9e(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__12b452541e5d3b1e3bca70f73f121afced3665878ee15e179799346351aefff2(
    value: typing.Mapping[builtins.str, builtins.str],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__699c745522b77730f7efd0081478c62be8f9b66ea42a06c900385149b4e1d72e(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__15ae42c0c38ee4c7b68c1ed04bcfa74e64b8c1bb1c780a5f07a3b9a4a8eed5e5(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__e6f010dd962f101be9896adf9347fba896da5bea0341b0852ccda31380df9685(
    value: typing.Optional[ServiceMetadata],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__03f320219ee5eea7af4f1ae3a7dbc7c2815442cfb1407e89cc3939597b51cb94(
    *,
    allocate_load_balancer_node_ports: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
    cluster_ip: typing.Optional[builtins.str] = None,
    cluster_ips: typing.Optional[typing.Sequence[builtins.str]] = None,
    external_ips: typing.Optional[typing.Sequence[builtins.str]] = None,
    external_name: typing.Optional[builtins.str] = None,
    external_traffic_policy: typing.Optional[builtins.str] = None,
    health_check_node_port: typing.Optional[jsii.Number] = None,
    internal_traffic_policy: typing.Optional[builtins.str] = None,
    ip_families: typing.Optional[typing.Sequence[builtins.str]] = None,
    ip_family_policy: typing.Optional[builtins.str] = None,
    load_balancer_class: typing.Optional[builtins.str] = None,
    load_balancer_ip: typing.Optional[builtins.str] = None,
    load_balancer_source_ranges: typing.Optional[typing.Sequence[builtins.str]] = None,
    port: typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.Sequence[typing.Union[ServiceSpecPort, typing.Dict[builtins.str, typing.Any]]]]] = None,
    publish_not_ready_addresses: typing.Optional[typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable]] = None,
    selector: typing.Optional[typing.Mapping[builtins.str, builtins.str]] = None,
    session_affinity: typing.Optional[builtins.str] = None,
    session_affinity_config: typing.Optional[typing.Union[ServiceSpecSessionAffinityConfig, typing.Dict[builtins.str, typing.Any]]] = None,
    type: typing.Optional[builtins.str] = None,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__ef17b13096e97c955edacd76e54ebada6a9004312f1d7ceca0c1625cb4fadee5(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__087915b9f41837b39fdb3d5efb9517a967fb502e6f5cd6c3bc6e72600907aeea(
    value: typing.Union[_cdktf_9a9027ec.IResolvable, typing.Sequence[typing.Union[ServiceSpecPort, typing.Dict[builtins.str, typing.Any]]]],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__a4f73a743eff3c7b3a15e87754d12ece5c76127ae555560400ff4e3c14a66ca9(
    value: typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__8e17ff5cde935e787a3ba9573e2bca2e5e2885b83006d57e0f1a00cdc1a16ae5(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__ffaec4b83256f07294f8a5ab40e8b4138fdc43ec53b864fbda4e846a978304aa(
    value: typing.List[builtins.str],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__0a61a18194e02df61d3cfbcdc9925d1d031828b0c356cea4e9ce21be9c9ef823(
    value: typing.List[builtins.str],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__c44d27c86046c4471bbdf5fc27b01e15ba8cf0bdbd355bcec7a03f747a9aa9b8(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__4f7cb1f153f104853fa730d0a89488a0626c253b244ead0e2254418cb1da67b6(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__9548c75677907311e1bd63b49a11cfcc0d59768524f440493ad5ad51df1036f6(
    value: jsii.Number,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__a1800ac2e733251918179c22a92cc9a02c3bfa44fd676897eebc5794a459d007(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__39b64d81b2a253bfce793b2d0bdfb92d3b269fa24d5b1ca38d18629ecb487e4d(
    value: typing.List[builtins.str],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__206a2f05bf3d96ab2ea50a3e617571e32506b850c953141c74f91ed89d8c59f6(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__89f619f97a822d4b20d2090225946836f88fe5e4af1b616046e2e88ae16b1adb(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__189dc7215e3d66cdc05cf1535dde9be5559dec91150fd22904bb300931e43da6(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__24ebbe4e5205fed5a3267f432d4e1dc7ea936f9da7bb33ef0ec086ed4a8d07b9(
    value: typing.List[builtins.str],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__4eafc425ec1b1caef5d5f56c6fae791c7c42109a430a960cc2bca071e19ca224(
    value: typing.Union[builtins.bool, _cdktf_9a9027ec.IResolvable],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__1cd7e0c4398f15e4b021f6f8f317496fa8714bdef02dc5074b7f7be34d004456(
    value: typing.Mapping[builtins.str, builtins.str],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__fa6243c723e96d5d2d31bfaa3e0bde1d40beac12141526c2ac84bf6dc9cbd3a6(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__1be0a835868627f60c65b090ab48a5ac1c1176eb0b82898defd4cb3a94730e61(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__7e13e85b739ca2f109d527dd676625db847d4c3d2a04098bece381af25c0d533(
    value: typing.Optional[ServiceSpec],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__dfa62c2f42e5704b799c2ab9b74fadaa889a38cc2bb43c1ebeb7fd24e4354481(
    *,
    port: jsii.Number,
    app_protocol: typing.Optional[builtins.str] = None,
    name: typing.Optional[builtins.str] = None,
    node_port: typing.Optional[jsii.Number] = None,
    protocol: typing.Optional[builtins.str] = None,
    target_port: typing.Optional[builtins.str] = None,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__14cbaf879a1ac09537a3f29f69184e9533cdc8949882d5c5d45a633159accf28(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
    wraps_set: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__7d405849f5e0dce5a1057d8efc88127ba9bfcbfd3cf140327638932c73bda135(
    index: jsii.Number,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__fc63ce87654afe099c1e7e30504e9dae9765df5510c8b98b94e5650a8c68e03b(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__61a17ba446ef217d9b7ad56c782c00610b462636522cd72866b542d783ebf3ba(
    value: _cdktf_9a9027ec.IInterpolatingParent,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__86b83c05a809d14c65ac7e9aea6958228d0a50ae00f45d0236637a46e8234c3e(
    value: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__36d4209fb9ca6b2eaf5ef1ef73238f18936f32a28e2c7f7ae69c5fdd954342e2(
    value: typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, typing.List[ServiceSpecPort]]],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__4525fd4094001093f1f89c9b67770297099d96afda372e1d9b00e4f0ec221079(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
    complex_object_index: jsii.Number,
    complex_object_is_from_set: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__83d4c9a221bf213c6ec6a3c74942c1f05b344ba1c8025dd479cc2b766013eabb(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__2ecc771ffde6b49da755506dca496777e78528f0bb6b1d86d2d1db9fab46a294(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__7a164de8289e6184e3ca54bfdda067a9f232b295a7178a7a4d266f09f08f6e06(
    value: jsii.Number,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__967caac236bbae765fea0b47ee18f7449deaa64e224c77daee73a23050146bf7(
    value: jsii.Number,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__9175f51d176ec09bb6c7ffd0c645b36d9f2f08e906739fa66e4bbdcb313759cc(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__aa467c462d554f6377c9315b470e518036c31fedd0384f904c24c2ef82bee15c(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__7ef6cd646d56464bfa46a6c821b3c761b9f762f737ff996bc199a42639477048(
    value: typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, ServiceSpecPort]],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__56cbf679d417c53baf9f20df11ca36d141bc94c346adef4803ae16d614a64cb6(
    *,
    client_ip: typing.Optional[typing.Union[ServiceSpecSessionAffinityConfigClientIp, typing.Dict[builtins.str, typing.Any]]] = None,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__ca5cb91d01776dd3876de5dd03af431c67c1cd2f99c884b7aa21d2af739d6a48(
    *,
    timeout_seconds: typing.Optional[jsii.Number] = None,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__29bb1b9a02cd3cd872f3ba26979b1f9ccc1da0ca8d4acd5b1431499de2701436(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__86c6de3b9dd402fe6fd3a40ce4b950e7e60e74efd49ee69c8e96074a9c895722(
    value: jsii.Number,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__e546be226f23a44979cee419329497fa4034a797ddd37e9c638a67fa385b8112(
    value: typing.Optional[ServiceSpecSessionAffinityConfigClientIp],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__373241708a7d5153ec2a6ef06a2d01dad4697869f2b13ac35e70b34716a078f5(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__3557db88b732f7218c9fe763db2f40388468f1e0b0b6f4c0f7d9497e615d4855(
    value: typing.Optional[ServiceSpecSessionAffinityConfig],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__65e34c91f40dd12b785184ff640f81256d717eac8132a0f7070c9a9a8ffdf907(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
    wraps_set: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__4ccd056e228d0359e708559fa6ac7b39f84a318233be854b8fb69ec344348ace(
    index: jsii.Number,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__80e416b8b7ee7aece9b574322a9e0e2bbd7b4d97baedb617a0cb8ac48427844f(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__267e98f31c7d4340b6df1eac29f0077da25a432f0e255aa978bdab0aa4bc5d76(
    value: _cdktf_9a9027ec.IInterpolatingParent,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__861247469de653396d1694235b3883b5dbbbf2e9c73e389aeb8dd68c538b6189(
    value: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__e60cc630f1394ad52d1989c2ec3c691e96ba49d38a654d0fa91bfb33fda84ccf(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
    wraps_set: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__28f6eb2a4afa45a81f2b82b4cdb4e6945188d798813540781f95149214b3e966(
    index: jsii.Number,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__41778733a629be32f76298ec99ef749e860f664c86c762b7bf8da2fd5917f20f(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__e42be57dc3bc0c449bb176ce88c3a06f22dae5311bdc11b8c9e97cba0512e841(
    value: _cdktf_9a9027ec.IInterpolatingParent,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__46b68cf4cdde48ab0042d240a5f14af5379607afd5453612a83f09e9396bc8f4(
    value: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__fcacbe5baead16373d11976b649428a2c57749ecc00b1a4f6c3f674eb67128f9(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
    complex_object_index: jsii.Number,
    complex_object_is_from_set: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__ca28450224ec38f89194f2b121396445b57df7d177430c70f1a5120e594f9287(
    value: typing.Optional[ServiceStatusLoadBalancerIngress],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__80bc58cca9608a2c356efbcc9571ddbdc2bd15980ec1d3e1dd070640117d14fa(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
    wraps_set: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__e90dd495578b6e2005d83f9bfc1219b5c1bf3fce6afd31eb88ddb291e4b29cba(
    index: jsii.Number,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__8d705a995ef12cf93668e9ffaef340696827d3ce452fe0ec245ce29f968676e5(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__e9d8f03b0c35d15e14e9c14d7bc9b14bbd62fd5398caa1ea2f677cfcaeb43316(
    value: _cdktf_9a9027ec.IInterpolatingParent,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__1ea0f3ecda6159d148b9e50604e0fc959d2a9718827e85aa4d4448d33996e974(
    value: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__496254de608e31c6b8c1198f2136fb23495f72c4bf00dde96e80e03aff490d67(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
    complex_object_index: jsii.Number,
    complex_object_is_from_set: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__7ad9a2c887f2889e37ab43bd14fe8bb0b64811d56ed7549aaacbbed42f1a40e1(
    value: typing.Optional[ServiceStatusLoadBalancer],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__e83b04ed2f4a3f18d905edcc9cbcc2f37d11afecffde28c00a8aca3517f37562(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
    complex_object_index: jsii.Number,
    complex_object_is_from_set: builtins.bool,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__ff5cc3c03637a9bdd641f4ed9621ed5a7c7ca504db30f17fe024973b44b6f70a(
    value: typing.Optional[ServiceStatus],
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__a00dd7b62182a7deaa5344460346ae116d8c1e2d34aa66e4afb3722f488930d4(
    *,
    create: typing.Optional[builtins.str] = None,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__c88e3c75267dbaba86985d4faf30e357e15bdcbc681a372dc79afbbe6b4ab07d(
    terraform_resource: _cdktf_9a9027ec.IInterpolatingParent,
    terraform_attribute: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__e57a48d12a6d923b6866e9ed2457e0814d69d8c577d3823f61c5f1df0d34b512(
    value: builtins.str,
) -> None:
    """Type checking stubs"""
    pass

def _typecheckingstub__989498eadfd8b1d4f89804fa3d3c0636d94443b0f4b2b284eeee21b4e6fee1d5(
    value: typing.Optional[typing.Union[_cdktf_9a9027ec.IResolvable, ServiceTimeouts]],
) -> None:
    """Type checking stubs"""
    pass