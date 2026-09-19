import logging
import random
import urllib.parse
from typing import Optional

import aiohttp
from open_webui.env import AIOHTTP_CLIENT_SESSION_SSL
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.session_pool import get_session
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)
default_headers = {'User-Agent': 'Mozilla/5.0'}


def _round_dimension(value: int, multiple: int = 16) -> int:
    multiple = max(1, int(multiple or 1))
    return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


def normalize_dimensions(width: int, height: int, max_pixels: Optional[int] = None, multiple_of: int = 16) -> tuple[int, int]:
    width, height = max(1, int(width)), max(1, int(height))
    if max_pixels and width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
    return _round_dimension(width, multiple_of), _round_dimension(height, multiple_of)


def parse_size(size: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if not size or 'x' not in size:
        return None, None
    try:
        width, height = (int(value) for value in size.lower().split('x', 1))
        return width, height
    except (TypeError, ValueError):
        return None, None


async def queue_prompt(prompt, client_id, base_url, api_key):
    session = await get_session()
    async with session.post(
        f'{base_url}/prompt',
        json={'prompt': prompt, 'client_id': client_id},
        headers={**default_headers, 'Authorization': f'Bearer {api_key}'},
        ssl=AIOHTTP_CLIENT_SESSION_SSL,
    ) as response:
        response.raise_for_status()
        return await response.json()


async def get_image(filename, subfolder, folder_type, base_url, api_key):
    values = urllib.parse.urlencode({'filename': filename, 'subfolder': subfolder, 'type': folder_type})
    session = await get_session()
    async with session.get(
        f'{base_url}/view?{values}',
        headers={**default_headers, 'Authorization': f'Bearer {api_key}'},
        ssl=AIOHTTP_CLIENT_SESSION_SSL,
    ) as response:
        response.raise_for_status()
        return await response.read()


def get_image_url(filename, subfolder, folder_type, base_url):
    values = urllib.parse.urlencode({'filename': filename, 'subfolder': subfolder, 'type': folder_type})
    return f'{base_url}/view?{values}'


async def get_history(prompt_id, base_url, api_key):
    session = await get_session()
    async with session.get(
        f'{base_url}/history/{prompt_id}',
        headers={**default_headers, 'Authorization': f'Bearer {api_key}'},
        ssl=AIOHTTP_CLIENT_SESSION_SSL,
    ) as response:
        response.raise_for_status()
        return await response.json()


async def _ws_get_images(ws, workflow, client_id, base_url, api_key):
    prompt_id = (await queue_prompt(workflow, client_id, base_url, api_key))['prompt_id']
    output_images = []
    async for message in ws:
        if message.type == aiohttp.WSMsgType.TEXT:
            payload = JSONCodec.loads(message.data)
            if payload['type'] == 'executing':
                data = payload['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            log.error('WebSocket closed unexpectedly: %s', message.type)
            break
    history = (await get_history(prompt_id, base_url, api_key))[prompt_id]
    for node_id, node_output in history['outputs'].items():
        if node_id in workflow and workflow[node_id].get('class_type') in ('SaveImage', 'PreviewImage'):
            for image in node_output.get('images', []):
                output_images.append({'url': get_image_url(image['filename'], image['subfolder'], image['type'], base_url)})
    return {'data': output_images}


async def comfyui_upload_image(image_file_item, base_url, api_key):
    _, (filename, file_bytes, mime_type) = image_file_item
    form = aiohttp.FormData()
    form.add_field('image', file_bytes, filename=filename, content_type=mime_type)
    form.add_field('type', 'input')
    session = await get_session()
    async with session.post(
        f'{base_url}/api/upload/image',
        data=form,
        headers={'Authorization': f'Bearer {api_key}'} if api_key else {},
        ssl=AIOHTTP_CLIENT_SESSION_SSL,
    ) as response:
        response.raise_for_status()
        return await response.json()


class ComfyUINodeInput(BaseModel):
    type: Optional[str] = None
    node_ids: list[str] = Field(default_factory=list)
    key: Optional[str] = 'text'
    value: Optional[str] = None


class ComfyUIWorkflow(BaseModel):
    workflow: str
    nodes: list[ComfyUINodeInput]


class ComfyUICreateImageForm(BaseModel):
    workflow: ComfyUIWorkflow
    prompt: str
    negative_prompt: Optional[str] = None
    width: int
    height: int
    size: Optional[str] = None
    max_pixels: Optional[int] = None
    multiple_of: int = 16
    n: int = 1
    steps: Optional[int] = None
    seed: Optional[int] = None

    @model_validator(mode='after')
    def normalize_size(self):
        size_width, size_height = parse_size(self.size)
        self.width, self.height = normalize_dimensions(
            size_width if size_width is not None else self.width,
            size_height if size_height is not None else self.height,
            self.max_pixels,
            self.multiple_of,
        )
        return self


class ComfyUIEditImageForm(BaseModel):
    workflow: ComfyUIWorkflow
    image: str | list[str]
    prompt: str
    width: Optional[int] = None
    height: Optional[int] = None
    size: Optional[str] = None
    max_pixels: Optional[int] = None
    multiple_of: int = 16
    n: Optional[int] = None
    steps: Optional[int] = None
    seed: Optional[int] = None

    @model_validator(mode='after')
    def normalize_size(self):
        size_width, size_height = parse_size(self.size)
        if size_width is not None and size_height is not None:
            self.width, self.height = normalize_dimensions(size_width, size_height, self.max_pixels, self.multiple_of)
        elif self.width is not None and self.height is not None:
            self.width, self.height = normalize_dimensions(self.width, self.height, self.max_pixels, self.multiple_of)
        return self


def _apply_workflow_nodes(workflow, nodes, model, payload):
    for node in nodes:
        if node.type:
            if node.type == 'model':
                for node_id in node.node_ids:
                    workflow[node_id]['inputs'][node.key] = model
            elif node.type == 'prompt':
                for node_id in node.node_ids:
                    workflow[node_id]['inputs'][node.key or 'text'] = payload.prompt
            elif node.type == 'negative_prompt':
                for node_id in node.node_ids:
                    workflow[node_id]['inputs'][node.key or 'text'] = payload.negative_prompt
            elif node.type == 'image':
                values = payload.image if isinstance(payload.image, list) else [payload.image]
                for index, node_id in enumerate(node.node_ids):
                    if index < len(values):
                        workflow[node_id]['inputs'][node.key] = values[index]
            elif node.type in ('width', 'height'):
                value = getattr(payload, node.type, None)
                if value is not None:
                    for node_id in node.node_ids:
                        workflow[node_id]['inputs'][node.key or node.type] = value
            elif node.type == 'n':
                for node_id in node.node_ids:
                    workflow[node_id]['inputs'][node.key or 'batch_size'] = payload.n
            elif node.type == 'steps':
                for node_id in node.node_ids:
                    workflow[node_id]['inputs'][node.key or 'steps'] = payload.steps
            elif node.type == 'seed':
                seed = payload.seed if payload.seed else random.randint(0, 1125899906842624)
                for node_id in node.node_ids:
                    workflow[node_id]['inputs'][node.key] = seed
        else:
            for node_id in node.node_ids:
                workflow[node_id]['inputs'][node.key] = node.value


def _set_workflow_dimensions(workflow, payload):
    if payload.width is None or payload.height is None:
        return
    for node in workflow.values():
        inputs = node.get('inputs', {}) if isinstance(node, dict) else {}
        if 'width' in inputs:
            inputs['width'] = payload.width
        if 'height' in inputs:
            inputs['height'] = payload.height


async def _run_workflow(model, payload, client_id, base_url, api_key):
    workflow = JSONCodec.loads(payload.workflow.workflow)
    _set_workflow_dimensions(workflow, payload)
    _apply_workflow_nodes(workflow, payload.workflow.nodes, model, payload)
    ws_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://')
    session = await get_session()
    async with session.ws_connect(
        f'{ws_url}/ws?clientId={client_id}',
        headers={'Authorization': f'Bearer {api_key}'},
        ssl=AIOHTTP_CLIENT_SESSION_SSL,
    ) as ws:
        return await _ws_get_images(ws, workflow, client_id, base_url, api_key)


async def comfyui_create_image(model: str, payload: ComfyUICreateImageForm, client_id, base_url, api_key):
    try:
        return await _run_workflow(model, payload, client_id, base_url, api_key)
    except Exception as error:
        log.exception('Error during image generation: %s', error)
        return None


async def comfyui_edit_image(model: str, payload: ComfyUIEditImageForm, client_id, base_url, api_key):
    try:
        return await _run_workflow(model, payload, client_id, base_url, api_key)
    except Exception as error:
        log.exception('Error during image editing: %s', error)
        return None
