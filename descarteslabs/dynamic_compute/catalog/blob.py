"""Funtionality around DL catalog blobs"""

from typing import Dict, List, Optional, Set, Union

import descarteslabs as dl

from ..compute_map import ComputeMap
from ..pyversions import PythonVersion


def get_blob_or_fail(
    id: Optional[str] = None,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> dl.catalog.Blob:
    """Gets a blob from dl catalog or fails if one isn't found. One of either
    id or name must be specified.

    Parameters
    ----------
    id : Optional[str], optional
        The id of the blob to retrieve, by default None
    name : Optional[str], optional
        The name of the blob to retrieve, by default None
    namespace : Optional[str], optional
        The namespace of the blob. Only required if providing name and the blob
        is not in the default namespace, by default None

    Returns
    -------
    dl.catalog.Blob
        The catalog blob object
    """

    if (not id and not name) or (id and name):
        err_msg = "Must specify exactly one of id or name parameters"
        raise ValueError(err_msg)

    blob = dl.catalog.Blob.get(id=id, name=name, namespace=namespace)

    if blob is None:
        err_msg = (
            f"Blob {id or name} either does not exist "
            "or you do not have access to it"
        )
        raise dl.exceptions.NotFoundError(err_msg)

    return blob


def create_blob_and_upload_data(
    data: str,
    name: str,
    namespace: Optional[str] = None,
    description: Optional[str] = None,
    extra_properties: Optional[Dict[str, Union[str, int, float]]] = None,
    readers: Optional[List[str]] = None,
    writers: Optional[List[str]] = None,
    storage_type: Optional[Union[str, dl.catalog.StorageType]] = None,
    tags: Optional[List[str]] = None,
) -> dl.catalog.Blob:
    """Creates a new blob and uploads data to it

    Parameters
    ----------
    data : str
        The data to upload to the blob
    name : str
        The name to give the blob
    namespace : Optional[str], optional
        The namespace to place the blob into, by default None
    description : Optional[str], optional
        A description of the blob, by default None
    extra_properties : Optional[dict[str, Union[str, int, float]]], optional
        A dictionary of extra properties for the blob, by default None
    readers : Optional[list[str]], optional
        A list of emails, orgs, or users to give read access, by default None
    writers : Optional[list[str]], optional
        A list of emails, orgs, or users to give write access, by default None
    storage_type : Optional[Union[str, dl.catalog.StorageType]]
        The storage type of this object, defaults to "data"
    tags: Optional[List[str]], optional
        A list of tags to assign to this blob for easy filtering later

    Returns
    -------
    dl.catalog.Blob
        The updated/created blob
    """

    blob = dl.catalog.Blob(
        name=name,
        namespace=namespace,
        description=description,
        extra_properties=extra_properties,
        readers=readers,
        writers=writers,
        storage_type=storage_type or dl.catalog.StorageType.DATA,
        tags=tags,
    )

    blob.upload_data(data)

    return blob


def find_blobs() -> dl.catalog.BlobSearch:
    """Searches for saved dynamic compute blobs and returns a BlobSearch object

    Returns
    -------
    dl.catalog.BlobSearch
    """

    return dl.catalog.Blob.search().filter(
        dl.catalog.properties.storage_type == "dyncomp"
    )


def print_blobs() -> None:
    """Prints the blob names of saved dynamic compute objects in catalog"""

    for blob in find_blobs():
        print(f"Type: {blob.extra_properties['graft_type']}, Id: {blob.id}")


def delete_blob(blob_id: str) -> None:
    """Deletes a saved dynamic compute object from catalog

    Parameters
    ----------
    blob_id : str
        The id of the blob to delete
    """

    get_blob_or_fail(blob_id).delete()


def share_blob(
    blob_id: str,
    emails: Optional[List[str]] = None,
    orgs: Optional[List[str]] = None,
    users: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    *,
    as_readers: bool = False,
    as_writers: bool = False,
):
    """Shares a saved dynamic compute object

    Parameters
    ----------
    blob_id : str
        The id of the blob to share
    emails : Optional[list[str]], optional
        A list of emails to share the blob with. Must be formatted like
        'email:jane.doe@place.com', by default None
    orgs : Optional[list[str]], optional
        A list of orgs to share the blob with. Must be formatted like
        'org:nameoforg' , by default None
    users : Optional[list[str]], optional
        A list of users to share the blob with. Must be formatted like
        'user:userid', by default None
    groups : Optional[list[str]], optional
        A list of groups to share the blob with. Must be formatted like
        'group:nameofgroup', by default None
    as_readers : bool, optional
        Whether to give the provided prinipals read access, by default False
    as_writers : bool, optional
        Whether to give the provided principals write access which also includes
        read access, by default False
    """

    def _update_blob_principals(blob: dl.catalog.Blob, attr: str, principals: Set[str]):
        for principal in principals:
            if principal not in getattr(blob, attr):
                getattr(blob, attr).append(principal)

    if not as_readers and not as_writers:
        err_msg = "At least one of 'as_readers' and 'as_writers' must be set to True"
        raise ValueError(err_msg)

    if not emails and not orgs and not users and not groups:
        err_msg = "At least one of 'emails', 'orgs', 'groups', or 'users' must be set"
        raise ValueError(err_msg)

    blob = get_blob_or_fail(blob_id)

    principals = set((emails or []) + (orgs or []) + (users or []) + (groups or []))

    if as_readers and not as_writers:
        _update_blob_principals(blob, "readers", principals)

    if as_writers:
        _update_blob_principals(blob, "writers", principals)

    blob.save()


def save_to_blob(
    blueprint: ComputeMap,
    name: str,
    namespace: Optional[str] = None,
    description: Optional[str] = None,
    extra_properties: Optional[Dict[str, Union[str, int, float]]] = None,
    readers: Optional[List[str]] = None,
    writers: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
):
    """Saves a dynamic compute object to catalog as a Blob

    Parameters
    ----------
    blueprint : Mosaic, ImageStack, or ImageStackGroupBy
        The dynamic compute object to save to a blob
    name : str
        The name to give the blob in catalog
    namespace : str
        An optional namespace to add to the default namespace. By default the
        namespace is set to {org}:{user_hash} and setting namespace here modifies
        that to {org}:{user_hash}:{namespace}
    description : Optional[str], optional
        A description of the blob, by default None
    extra_properties : Optional[dict[str, Union[str, int, float]]], optional
        Any extra properties to be stored in the blob, by default None
    readers : Optional[list[str]], optional
        A list of emails, orgs, groups, and users to give read access to the blob, by default None
    writers : Optional[list[str]], optional
        A list of emails, orgs, groups, and users to give write access to the blob, by default None
    tags : Optional[List[str]], optional
        A list of tags to assign to the blob

    Returns
    -------
    str
        The id of the blob created
    """

    default_namespace = dl.catalog.Blob.namespace_id(None)
    if namespace:
        default_namespace += f":{namespace}"

    py_version = PythonVersion.from_sys()

    extra_properties = extra_properties or {}
    extra_properties["graft_type"] = blueprint.__class__.__name__
    extra_properties["python_version"] = py_version.major_minor

    blob = create_blob_and_upload_data(
        blueprint.serialize(),
        name,
        namespace=default_namespace,
        description=description,
        extra_properties=extra_properties,
        readers=readers,
        writers=writers,
        storage_type=dl.catalog.StorageType.DYNCOMP,
        tags=tags,
    )

    return blob.id


def _raise_for_incompatible_python_version(saved_version: str):
    """Raises a RuntimeError if system python version is incompatible with
    saved_version

    Parameters
    ----------
    saved_version : str
        The saved python version in the blob

    Raises
    ------
    RuntimeError
        Raised when system and saved python versions are incompatible
    """

    sys_version = PythonVersion.from_sys()
    saved_version = PythonVersion.from_string(saved_version)
    if not sys_version.compatible_with(saved_version):
        err_msg = (
            "Dynamic compute blob incompatible with current python version"
            f" '{sys_version.major_minor}', may only be loaded with "
            f"python version '{saved_version.major_minor}'"
        )
        raise RuntimeError(err_msg)


def load_from_blob(blob_id: str) -> ComputeMap:
    """Loads an dynamic compute type from a catalog blob

    Parameters
    ----------
    blob_id : str
        The id of the blob to load

    Returns
    -------
    ImageStack, Mosaic, or ImageStackGroupBy
        The loaded dynamic compute object
    """

    blob = get_blob_or_fail(blob_id)

    _raise_for_incompatible_python_version(blob.extra_properties["python_version"])

    return ComputeMap.__SUBCLASSES__[blob.extra_properties["graft_type"]].deserialize(
        blob.data()
    )
