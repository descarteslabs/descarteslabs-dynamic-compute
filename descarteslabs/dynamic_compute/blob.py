"""Funtionality around DL catalog blobs"""

import json
from typing import Dict, List, Optional, Set, Union

import descarteslabs as dl

GRAFTS_NAMESPACE = "dynamic-compute"


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
        raise TypeError(err_msg)

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
        tags=tags,
    )

    blob.upload_data(data)

    return blob


def load_graft_from_blob(name: str, graft_type: str) -> Dict:
    """Loads a graft from a dl.catalog Blob

    Parameters
    ----------
    name : str
        The name of the saved graft
    graft_type : str
        The ComputeMap subclass this graft is trying to be loaded into

    Returns
    -------
    Dict
        The loaded graft
    """

    blob = get_blob_or_fail(name=name, namespace=GRAFTS_NAMESPACE)
    if blob.extra_properties["graft_type"] != graft_type:
        err_msg = f"Retrieved blob is not compatible with type '{graft_type}'"
        raise dl.exceptions.ConflictError(err_msg)

    return json.loads(blob.data())


def list_catalog_blobs(
    *, return_blobs: bool = False
) -> Optional[List[dl.catalog.Blob]]:
    """Prints the blob names of saved ImageStacks or Mosaics in catalog

    Parameters
    ----------
    return_blobs : Optional[bool], optional
        Whether or not to return the found blobs, by default False

    Returns
    -------
    Optional[list[dl.catalog.Blob]]
        The blobs found or None
    """

    blobs = dl.catalog.Blob.search().filter(
        dl.catalog.properties.namespace == GRAFTS_NAMESPACE
    )
    for blob in blobs:
        print(f"Type: {blob.extra_properties['graft_type']}, Name: {blob.name}")

    if return_blobs:
        return list(blobs)


def delete_blob_in_catalog(name: str) -> None:
    """Deletes a saved ImageStack or Mosaic from catalog

    Parameters
    ----------
    name : str
        The name of the blob to delete
    """

    get_blob_or_fail(name=name, namespace=GRAFTS_NAMESPACE).delete()


def share_catalog_blob(
    name: str,
    emails: Optional[List[str]] = None,
    orgs: Optional[List[str]] = None,
    users: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    *,
    as_readers: bool = False,
    as_writers: bool = False,
):
    """Shares a saved ImageStack or Mosaic

    Parameters
    ----------
    name : str
        The name of the blob to share
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
        Whether to give the provided principals write access, by default False
    """

    def _update_blob_principals(blob: dl.catalog.Blob, attr: str, principals: Set[str]):
        for principal in principals:
            if principal not in getattr(blob, attr):
                getattr(blob, attr).append(principal)

    if not as_readers and not as_writers:
        err_msg = (
            "At least one or both 'as_readers' and 'as_writers' must be set to True"
        )
        raise ValueError(err_msg)

    if not emails and not orgs and not users and not groups:
        err_msg = "At least one of 'emails', 'orgs', 'groups', or 'users' must be set"
        raise ValueError(err_msg)

    blob = get_blob_or_fail(name=name, namespace=GRAFTS_NAMESPACE)

    principals = set((emails or []) + (orgs or []) + (users or []) + (groups or []))

    if as_readers:
        _update_blob_principals(blob, "readers", principals)

    if as_writers:
        _update_blob_principals(blob, "writers", principals)

    blob.save()
