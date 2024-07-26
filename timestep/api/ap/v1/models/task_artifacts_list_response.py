from datetime import date, datetime  # noqa: F401
from typing import Dict, List  # noqa: F401

from timestep.api.ap.v1 import util
from timestep.api.ap.v1.models.artifact import Artifact  # noqa: E501
from timestep.api.ap.v1.models.base_model import Model
from timestep.api.ap.v1.models.pagination import Pagination  # noqa: E501


class TaskArtifactsListResponse(Model):
    """NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).

    Do not edit the class manually.
    """

    def __init__(self, artifacts=None, pagination=None):  # noqa: E501
        """TaskArtifactsListResponse - a model defined in OpenAPI

        :param artifacts: The artifacts of this TaskArtifactsListResponse.  # noqa: E501
        :type artifacts: List[Artifact]
        :param pagination: The pagination of this TaskArtifactsListResponse.  # noqa: E501
        :type pagination: Pagination
        """
        self.openapi_types = {
            'artifacts': List[Artifact],
            'pagination': Pagination
        }

        self.attribute_map = {
            'artifacts': 'artifacts',
            'pagination': 'pagination'
        }

        self._artifacts = artifacts
        self._pagination = pagination

    @classmethod
    def from_dict(cls, dikt) -> 'TaskArtifactsListResponse':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The TaskArtifactsListResponse of this TaskArtifactsListResponse.  # noqa: E501
        :rtype: TaskArtifactsListResponse
        """
        return util.deserialize_model(dikt, cls)

    @property
    def artifacts(self) -> List[Artifact]:
        """Gets the artifacts of this TaskArtifactsListResponse.


        :return: The artifacts of this TaskArtifactsListResponse.
        :rtype: List[Artifact]
        """
        return self._artifacts

    @artifacts.setter
    def artifacts(self, artifacts: List[Artifact]):
        """Sets the artifacts of this TaskArtifactsListResponse.


        :param artifacts: The artifacts of this TaskArtifactsListResponse.
        :type artifacts: List[Artifact]
        """
        if artifacts is None:
            raise ValueError("Invalid value for `artifacts`, must not be `None`")  # noqa: E501

        self._artifacts = artifacts

    @property
    def pagination(self) -> Pagination:
        """Gets the pagination of this TaskArtifactsListResponse.


        :return: The pagination of this TaskArtifactsListResponse.
        :rtype: Pagination
        """
        return self._pagination

    @pagination.setter
    def pagination(self, pagination: Pagination):
        """Sets the pagination of this TaskArtifactsListResponse.


        :param pagination: The pagination of this TaskArtifactsListResponse.
        :type pagination: Pagination
        """
        if pagination is None:
            raise ValueError("Invalid value for `pagination`, must not be `None`")  # noqa: E501

        self._pagination = pagination