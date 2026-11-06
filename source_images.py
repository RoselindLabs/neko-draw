"""Resolve real images inside one MaiBot chat; never turn descriptions into inputs."""
import base64
import io
import time
import warnings
from collections import OrderedDict

from PIL import Image
try:
    from .resource_budget import image_work
    from .result_images import validate_gif
except ImportError:  # Standalone offline tests.
    from resource_budget import image_work
    from result_images import validate_gif

MAX_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 4
MAX_PIXELS = 40_000_000


class MissingSourceImages(ValueError):
    """A valid lookup has no readable source; not a decode/limit failure."""


async def resolve_context_image(ctx, stream_id, source_id, index=0):
    """One explicitly selected context image; never substitute another index."""
    if not stream_id or not isinstance(source_id, str) or not source_id.strip():
        raise ValueError('上下文取图需要当前聊天和真实图片来源编号')
    if type(index) is not int or index < 0:
        raise ValueError('source_image_index 必须是非负整数，从 0 开始')
    result = await ctx.call_capability('maisaka.context.resolve_image',
                                       stream_id=stream_id, source_id=source_id.strip(), index=index)
    if not isinstance(result, dict) or result.get('success') is not True:
        raise ValueError('上下文图片不可读取，请补发或改用真实消息 ID')
    image = result.get('image')
    if not isinstance(image, dict):
        raise ValueError('宿主未返回上下文图片字节')
    # Reuse the same bounded Base64, pixel, format and animation validation.
    images = await image_work(extract, {'message_segments': [dict(image, type='image')]})
    if len(images) != 1:
        raise ValueError('宿主未返回一张可读取的上下文图片')
    return images


async def resolve_edit_images(ctx, stream_id, source_id='', current=None, cache=None, index=None):
    if index is not None:
        # Explicit selection is authoritative: an invalid index must not turn
        # into all images / another cached picture / a quote fallback.
        return await resolve_context_image(ctx, stream_id, source_id, index)
    try:
        return await resolve_images(ctx, stream_id, source_id, current, cache)
    except MissingSourceImages:
        if not source_id:
            raise  # Commands preserve their current-message/quote semantics.
        # Short aliases and tool media IDs do not necessarily exist in the DB.
        # Preserve legacy all-image/sticker/quote behaviour when it succeeds.
        return await resolve_context_image(ctx, stream_id, source_id)


class ImageCache:
    def __init__(self):
        self.entries = OrderedDict()

    def prune(self):
        now = time.monotonic()
        for key, (expires, _) in list(self.entries.items()):
            if expires <= now:
                del self.entries[key]
        total = sum(sum(map(len, images)) for _, images in self.entries.values())
        while self.entries and (total > 32 * 1024 * 1024 or len(self.entries) > 40):
            _, (_, images) = self.entries.popitem(last=False)
            total -= sum(map(len, images))

    def put(self, stream, message, images=None):
        mid = str(message.get('message_id') or '')
        images = extract(message) if images is None else images
        if mid and images:
            self.entries[(stream, mid)] = (time.monotonic() + 1800, images)
            self.entries.move_to_end((stream, mid))
        self.prune()

    def get(self, stream, mid):
        self.prune()
        return list(self.entries.get((stream, mid), (0, []))[1])


def mime_type(data):
    if data.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    raise ValueError('源图需要 PNG、JPEG、WebP 或 GIF 格式')


def normalize_source_image(data):
    """Validate image/sticker pixels; preserve fully validated GIF bytes.

    Never interpret emoji descriptions, Unicode faces or QQ face IDs as images.
    Other animations still use their first frame (skip APNG's separate poster).
    """
    if not data or len(data) > MAX_BYTES:
        raise ValueError('单张源图不能为空或超过 10 MiB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {'PNG', 'JPEG', 'WEBP', 'GIF'}:
                    raise ValueError('源图需要 PNG、JPEG、WebP 或 GIF 格式')
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('源图不能超过 4000 万像素')
                if image.format == 'GIF':
                    validate_gif(data, image)
                    return data
                if getattr(image, 'is_animated', False):
                    image.seek(1 if image.format == 'PNG' and image.info.get('default_image') else 0)
                    output = io.BytesIO()
                    image.convert('RGBA').save(output, format='PNG')
                    result = output.getvalue()
                    if len(result) > MAX_BYTES:
                        raise ValueError('动图首帧转换后超过 10 MiB')
                    return result
                image.load()  # verify() alone does not decode JPEG pixels.
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
                return data
    except (OSError, SyntaxError, EOFError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('原始图片无效或无法解码') from exc


def unwrap(value):
    if isinstance(value, list):
        return [m for v in value for m in unwrap(v)]
    if not isinstance(value, dict):
        return []
    if 'message_segments' in value or 'raw_message' in value:
        return [value]
    for key in ('message', 'messages', 'result'):
        if key in value:
            return unwrap(value[key])
    return []


def segments(message):
    return [s for key in ('message_segments', 'raw_message')
            for s in (message.get(key) or []) if isinstance(s, dict)]


def extract(message):
    images = []
    for segment in segments(message):
        if segment.get('type') not in ('image', 'emoji'):
            continue
        for source in (segment, segment.get('data')):
            if not isinstance(source, dict):
                continue
            raw = next((source.get(k) for k in ('binary_data_base64', 'base64', 'image_base64', 'file_base64') if source.get(k)), '')
            if not isinstance(raw, str) or not raw:
                continue
            if len(raw) > MAX_BYTES * 4 // 3 + 1024:
                raise ValueError('单张源图不能超过 10 MiB')
            if raw.startswith('data:image/'):
                raw = raw.split(',', 1)[-1]
            data = base64.b64decode(''.join(raw.split()), validate=True)
            if len(data) > MAX_BYTES:
                raise ValueError('单张源图不能超过 10 MiB')
            data = normalize_source_image(data)
            if data not in images:
                images.append(data)
            if len(images) > MAX_IMAGES:
                raise ValueError('一次最多使用 4 张源图')
    return images


async def resolve_images(ctx, stream_id, message_id='', current=None, cache=None):
    seen = set()
    own_images = {}

    async def lookup(mid, depth=0):
        mid = str(mid or '').strip()
        if not mid or mid in seen or depth > 2:
            return []
        if len(seen) >= 16:
            raise ValueError('引用消息过多，一次最多读取 16 条消息')
        seen.add(mid)
        cached = cache.get(stream_id, mid) if cache else []
        try:
            result = await ctx.call_capability('message.get_by_id', chat_id=stream_id,
                                              message_id=mid, include_binary_data=True)
        except Exception:
            if cached:
                return list(cached)  # Only this chat's exact message cache.
            raise
        messages = unwrap(result)
        images = await collect(messages[0], depth) if messages else list(cached)
        if len(images) > MAX_IMAGES:
            raise ValueError('一次最多使用 4 张源图')
        return images

    async def collect(message, depth=0, authoritative=True):
        mid = str(message.get('message_id') or '')
        if not authoritative and mid in own_images:
            images = list(own_images[mid])
        else:
            images = await image_work(extract, message)
            if images:
                # Host may have compressed the image since the receive hook.
                # Prefer fresh bytes rather than supplying two versions.
                if authoritative and mid:
                    own_images[mid] = list(images)
                    if cache:
                        cache.put(stream_id, message, images=images)
            elif cache:
                images = cache.get(stream_id, mid)
        refs = [message.get(k) for k in ('reply_to', 'reply_message_id', 'quote_message_id', 'quoted_message_id')]
        for key in ('reply', 'quote', 'quoted_message'):
            ref = message.get(key)
            if isinstance(ref, dict):
                ref = ref.get('message_id') or ref.get('id') or ref.get('target_message_id')
            refs.append(ref)
        for segment in segments(message):
            if segment.get('type') in ('reply', 'quote'):
                data = segment.get('data')
                refs.append((data.get('message_id') or data.get('id') or data.get('target_message_id')) if isinstance(data, dict) else data)
        for ref in refs:
            if isinstance(ref, (str, int)):
                for image in await lookup(ref, depth + 1):
                    if image not in images:
                        images.append(image)
                    if len(images) > MAX_IMAGES:
                        raise ValueError('一次最多使用 4 张源图')
        if len(images) > MAX_IMAGES:
            raise ValueError('一次最多使用 4 张源图')
        return images

    # Explicit IDs must not silently fall back to unrelated recent pictures.
    if message_id:
        images = await lookup(message_id)
    elif isinstance(current, dict):
        # Restore this message before collecting its quotes. A successful quote
        # must not suppress retrieval of the stripped current image.
        images = await lookup(current.get('message_id'))
        for image in await collect(current, authoritative=False):
            if image not in images:
                images.append(image)
    else:
        images = []
    if len(images) > MAX_IMAGES:
        raise ValueError('一次最多使用 4 张源图')
    if not images:
        raise MissingSourceImages('指定消息或引用里没有可读取的图片或表情包原图。请重新发送图片/表情包，并引用它使用 /draw edit <修改要求>')
    return images
