import json
import logging
import zipfile
from io import BytesIO, IOBase, StringIO

from data_export.json_conversion import RLE_To_JSON
from data_export.models import DataExport
from django.http import QueryDict
from projects.models import Project
from tasks.models import Task

logger = logging.getLogger(__name__)


def generate_export_file_for_coco(
    project: Project, tasks: list[Task], export_type: str, download_resources: bool, get_args: QueryDict
) -> tuple[IOBase, str, str]:
    # TODO: settings.EXPORT_DATA_SERIALIZER を独自のものに設定する方がキレイかも
    # NOTE: label_studio_converter が 0.0.58 時点では coco format を以下の理由でうまく取り扱えないので、一度 JSON で取得してから独自に変換する
    # - brushlabel に対応してない
    #   - https://github.com/HumanSignal/label-studio-converter/blob/0.0.58/label_studio_converter/converter.py#L654-L698
    if export_type != 'COCO':
        raise ValueError(f'Unsupported export type for generate_export_file_for_coco: {export_type}')

    # まずJSONフォーマットでデータを取得
    export_stream, content_type, filename = DataExport.generate_export_file(
        project, tasks, 'JSON', download_resources, get_args
    )
    
    # JSONデータを読み込む
    # ファイルポインタを先頭に戻す
    export_stream.seek(0)
    
    # ファイルタイプによって適切に読み込み
    if isinstance(export_stream, BytesIO):
        json_data = json.loads(export_stream.read().decode('utf-8'))
    else:
        json_data = json.load(export_stream)
    
    # COCO形式に変換
    coco_data = RLE_To_JSON(json_data)
    coco_data_str = json.dumps(coco_data, ensure_ascii=False)
    
    # BytesIOとして返す
    buffer = BytesIO(coco_data_str.encode('utf-8'))
    # ファイル名をCOCO形式用に変更
    filename = filename.replace('.json', '_coco.json')
    
    return buffer, content_type, filename


def get_json_str_from_zip(zip_bytes: bytes, original_filename) -> tuple[IOBase, str, str]:
    json_file_inside_zip = 'result.json'  # default name of label studio

    content_type = 'application/json'
    filename = original_filename.replace('.zip', '.json')

    with zipfile.ZipFile(BytesIO(zip_bytes)) as zip_file:
        if json_file_inside_zip in zip_file.namelist():
            file_contents = zip_file.read(json_file_inside_zip)
            
            # BytesIO を使用してバイトデータを返す
            export_stream = BytesIO(file_contents)
            return export_stream, content_type, filename
        else:
            raise FileNotFoundError(f'{json_file_inside_zip} not found in the zip archive')


def format_coco(data: IOBase) -> IOBase:
    # reset the file pointer to the beginning
    data.seek(0)
    
    if isinstance(data, BytesIO):
        coco_data = json.loads(data.read().decode('utf-8'))
    else:
        coco_data = json.load(data)

    # sort categories by id
    old_categories = coco_data.get('categories', [])
    sorted_categories = sorted(old_categories, key=lambda x: x['id'])

    # set path and file_name
    for image in coco_data.get('images', []):
        if 'file_name' in image:
            image['path'] = image['file_name']
            image['file_name'] = image['file_name'].split('/')[-1]

    coco_data['categories'] = sorted_categories

    # JSONとしてダンプし、バイト形式で返す
    return BytesIO(json.dumps(coco_data, ensure_ascii=False).encode('utf-8'))