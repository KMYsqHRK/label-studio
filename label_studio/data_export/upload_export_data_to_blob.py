import io
import logging
import posixpath

from azure.storage.blob import BlobType, ContentSettings
from io_storages.azure_blob.models import AzureBlobImportStorage
from projects.models import Project

logger = logging.getLogger(__name__)


# NOTE: SENSYN の現状の使い方だと coco format は blob にも保存された方が良いので、そっちにも保存する事にしてる
# - https://sensyn-robotics.slack.com/archives/CSQK3S8UQ/p1711613405519439?thread_ts=1711603464.160269&cid=CSQK3S8UQ
# - 画像も一緒に export しようとしてるけど、cloud storage (azure の場合 filename が azure-blob://... になる) に対応できてない
#   - https://github.com/HumanSignal/label-studio-converter/blob/0.0.58/label_studio_converter/utils.py#L134
def upload(project: Project, data, content_type: str, filename: str):
    # lable studio側の実装でDataExport.generate_export_fileからapplication/.jsonが返ってくるので、application/jsonに修正する
    # https://github.com/HumanSignal/label-studio/blob/9355c9c307ec49ec9bf89bc9036d7b06fb83e93f/label_studio/data_export/models.py#L167
    if content_type == 'application/.json':
        content_type = 'application/json'

    if content_type == 'application/json':
        # 様々な入力タイプに対応
        if isinstance(data, io.StringIO):
            data = data.getvalue()
        elif isinstance(data, io.BytesIO):
            # BytesIOの場合、バイトデータを文字列に変換
            data.seek(0)
            data = data.read().decode('utf-8')
        elif isinstance(data, str):
            # 既に文字列の場合はそのまま使用
            pass
        elif isinstance(data, bytes):
            # バイトデータの場合は文字列に変換
            data = data.decode('utf-8')
        else:
            logger.error(f"Unexpected data type: {type(data)}")
            raise TypeError(f"Expected StringIO, BytesIO, str, or bytes, got {type(data)}")
    else:
        raise ValueError(f'Unsupported content type for upload_export_data_to_blob: {content_type}')

    # only export if project has import blob setting
    if not hasattr(project, 'io_storages_azureblobimportstorages'):
        return

    azure_blob: AzureBlobImportStorage | None = project.io_storages_azureblobimportstorages.first()
    if azure_blob is None:
        return

    account_name = azure_blob.get_account_name()
    if not isinstance(account_name, str):
        return

    # Create a BlobServiceClient object using the connection string
    _, container_client = azure_blob.get_client_and_container()

    prefix = ''
    if azure_blob.prefix is not None:
        prefix = str(azure_blob.prefix)

    blob_name = posixpath.join(prefix, '.LS_exports', filename)
    blob_client = container_client.get_blob_client(blob_name)

    logger.info(f'upload export data to {blob_name} in {account_name}')
    blob_client.upload_blob(
        data, blob_type=BlobType.BLOCKBLOB, overwrite=True, content_settings=ContentSettings(content_type=content_type)
    )