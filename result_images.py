"""Bound API responses and validate generated image bytes before delivery."""
import base64
import io
import warnings

from PIL import Image

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_GIF_FRAMES = 200
MAX_GIF_DECODE_PIXELS = 40_000_000


def validate_gif(data, image):
    """Bound structure before decoding every frame; keep the original bytes.

    Pillow's seek EOF can also mean a truncated later frame. Count complete
    image blocks first, then require each of those frames to actually decode.
    Do not use n_frames (it scans the whole file before enforcing our budget).
    Called only inside the existing image_work slot by production callers.
    """
    position = 0

    def take(size):
        nonlocal position
        end = position + size
        if end > len(data):
            raise ValueError('GIF 数据截断')
        value = data[position:end]
        position = end
        return value

    def subblocks():
        while True:
            size = take(1)[0]
            if not size:
                return
            take(size)

    if take(6) not in (b'GIF87a', b'GIF89a'):
        raise ValueError('GIF 文件头无效')
    screen = take(7)
    width = int.from_bytes(screen[:2], 'little')
    height = int.from_bytes(screen[2:4], 'little')
    if not width or not height:
        raise ValueError('GIF 尺寸无效')
    if screen[4] & 128:
        take(3 * (2 ** ((screen[4] & 7) + 1)))
    frames = 0
    pixels = 0
    while True:
        marker = take(1)[0]
        if marker == 0x3B:
            if not frames or position != len(data):
                raise ValueError('GIF 结束标记无效')
            break
        if marker == 0x21:
            take(1)  # Extension label; content is a sub-block chain.
            subblocks()
        elif marker == 0x2C:
            descriptor = take(9)
            left = int.from_bytes(descriptor[:2], 'little')
            top = int.from_bytes(descriptor[2:4], 'little')
            fw = int.from_bytes(descriptor[4:6], 'little')
            fh = int.from_bytes(descriptor[6:8], 'little')
            if not fw or not fh or left + fw > width or top + fh > height:
                raise ValueError('GIF 帧尺寸无效')
            frames += 1
            pixels += width * height  # Composited canvas, not just changed rectangle.
            if frames > MAX_GIF_FRAMES or pixels > MAX_GIF_DECODE_PIXELS:
                raise ValueError('GIF 超过帧数或累计解码像素预算')
            if descriptor[8] & 128:
                take(3 * (2 ** ((descriptor[8] & 7) + 1)))
            if not 2 <= take(1)[0] <= 8:
                raise ValueError('GIF LZW 编码无效')
            subblocks()
        else:
            raise ValueError('GIF 数据块无效')
    for frame in range(frames):
        image.seek(frame)
        image.load()


async def read_response(response) -> bytes:
    result = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        if len(result) + len(chunk) > MAX_RESPONSE_BYTES:
            raise ValueError("生图接口响应超过 32 MiB")
        result.extend(chunk)
    return bytes(result)


def validate_image(data: bytes) -> bytes:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("生成图片为空或超过 16 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP", "GIF"}:
                    raise ValueError("生成图片必须是 PNG、JPEG、WebP 或 GIF")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("生成图片像素数量过大")
                if image.format == 'GIF':
                    validate_gif(data, image)
                    return data
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
    except (OSError, SyntaxError, EOFError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("接口返回的图片无效") from exc
    return data


def decode_image(raw: str) -> bytes:
    if len(raw) > (MAX_IMAGE_BYTES + 2) // 3 * 4 + 1024:
        raise ValueError("生成图片 Base64 超过大小限制")
    return validate_image(base64.b64decode("".join(raw.split()), validate=True))
