"""Synthetic GIF round trips only: no supplier requests or QQ sends."""
import base64
import io
import json
import types
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

import result_images
from source_images import extract, mime_type, normalize_source_image
import test_admission_safety as fixtures


def gif():
    output = io.BytesIO()
    frames = [Image.new('RGB', (3, 2), color) for color in ('red', 'blue', 'green')]
    frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:],
                   duration=[40, 120, 230], loop=3, disposal=2)
    return output.getvalue()


def bad_frame():
    data = bytearray(gif())
    with Image.open(io.BytesIO(data)) as image:
        image.seek(2)
        offset = image.tile[0][2]
    # Keep the descriptor, sub-block length and trailer intact; corrupt only
    # the third frame's LZW stream, after two valid frames.
    size = data[offset]
    data[offset + 1:offset + 1 + size] = b'\xff' * size
    return bytes(data)


class GifValidation(unittest.TestCase):
    def test_original_bytes_and_animation_metadata_survive_input_and_output(self):
        data = gif()
        for kind in ('image', 'emoji'):
            message = {'raw_message': [{'type': kind, 'binary_data_base64': base64.b64encode(data).decode()}]}
            self.assertEqual(extract(message), [data])
        self.assertEqual(normalize_source_image(data), data)
        self.assertEqual(result_images.validate_image(data), data)
        self.assertEqual(mime_type(data), 'image/gif')
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.info['loop'], 3)
            self.assertEqual(image.n_frames, 3)
            durations = []
            colors = []
            for frame in range(3):
                image.seek(frame)
                durations.append(image.info['duration'])
                colors.append(image.convert('RGB').getpixel((0, 0)))
            self.assertEqual(durations, [40, 120, 230])
            self.assertEqual(colors, [(255, 0, 0), (0, 0, 255), (0, 128, 0)])

    def test_gif87a_static_bytes_supported(self):
        output = io.BytesIO()
        Image.new('RGB', (1, 1)).save(output, format='GIF')
        data = output.getvalue()
        self.assertTrue(data.startswith(b'GIF87a'))
        self.assertEqual(normalize_source_image(data), data)
        self.assertEqual(result_images.validate_image(data), data)
        self.assertEqual(mime_type(data), 'image/gif')

    def test_later_corrupt_frame_and_truncation_rejected(self):
        for data in (bad_frame(), gif()[:-1], gif()[:-5], gif()[:-20]):
            for validate in (normalize_source_image, result_images.validate_image):
                with self.subTest(length=len(data), validator=validate.__name__), self.assertRaises(ValueError):
                    validate(data)

    def test_frame_and_composited_pixel_limits_checked_before_decode(self):
        data = gif()
        for validate in (normalize_source_image, result_images.validate_image):
            for constant, limit in (('MAX_GIF_FRAMES', 2), ('MAX_GIF_DECODE_PIXELS', 17)):
                with patch.object(result_images, constant, limit), patch.object(Image.Image, 'load', side_effect=AssertionError('must reject before decoding')):
                    with self.assertRaisesRegex(ValueError, '预算'):
                        validate(data)
            with patch.object(result_images, 'MAX_GIF_FRAMES', 3), patch.object(result_images, 'MAX_GIF_DECODE_PIXELS', 18):
                self.assertEqual(validate(data), data)


class GifRoutes(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.Safety.asyncSetUp
    post = fixtures.Safety.post

    async def test_multipart_chat_and_gemini_keep_gif_mime_and_bytes(self):
        data = gif()
        encoded = base64.b64encode(data).decode()
        post = self.post(fixtures.Response(json.dumps({'data': [{'b64_json': encoded}]}).encode()))
        self.assertEqual(await self.bot._openai_images('edit', self.bot.config.openai, [data]), data)
        field = post.call_args.kwargs['data']._fields[-1]
        self.assertEqual(field[0]['filename'], 'source-0.gif')
        self.assertEqual(field[1]['Content-Type'], 'image/gif')
        self.assertEqual(field[2], data)
        self.bot._post_json = AsyncMock(return_value={'choices': [{'message': {'content': 'data:image/gif;base64,' + encoded}}]})
        self.assertEqual(await self.bot._openai_chat('edit', self.bot.config.openai, [data]), data)
        content = self.bot._post_json.call_args.args[1]['messages'][0]['content']
        self.assertEqual(content[1]['image_url']['url'], 'data:image/gif;base64,' + encoded)
        self.bot._post_json.return_value = {'candidates': [{'content': {'parts': [{'inlineData': {'data': encoded}}]}}]}
        self.assertEqual(await self.bot._gemini_native('edit', [data]), data)
        parts = self.bot._post_json.call_args.args[1]['contents'][0]['parts']
        self.assertEqual(parts[1]['inlineData'], {'mimeType': 'image/gif', 'data': encoded})

    async def test_base64_data_url_chat_shapes_download_and_final_send(self):
        data = gif()
        encoded = base64.b64encode(data).decode()
        url = 'https://synthetic.invalid/animation.gif'
        data_url = 'data:image/gif;base64,' + encoded
        module = fixtures.mod
        client = AsyncMock()
        with patch.object(module, 'PublicClient', return_value=client), patch.object(module, 'local_addresses', return_value=set()), patch.object(module, 'download_public', new_callable=AsyncMock, return_value=(data, 'image/gif', url)) as download:
            for value in (encoded, data_url, url):
                self.assertEqual(await self.bot._bytes_from_reference(value, ''), data)
                self.assertEqual(await self.bot._openai_result({'data': [{'b64_json': value}]}, ''), data)
            self.assertEqual(download.await_count, 2)
            self.assertEqual(download.call_args.kwargs['max_image_bytes'], result_images.MAX_IMAGE_BYTES)
        for content in (data_url, f'![gif]({data_url})', [{'image_url': {'url': data_url}}], [{'data': encoded}], [{'b64_json': encoded}]):
            self.assertEqual(await self.bot._openai_result({'choices': [{'message': {'content': content}}]}, ''), data)
        self.assertEqual(await self.bot._openai_result({'choices': [{'message': {'images': [{'url': data_url}]}}]}, ''), data)
        self.bot.ctx.send = types.SimpleNamespace(image=AsyncMock(return_value=True))
        await self.bot._send(data, 'chat', 'animation')
        args = self.bot.ctx.send.image.call_args
        self.assertEqual(base64.b64decode(args.args[0]), data)
        self.assertEqual(args.args[1], 'chat')
        self.assertTrue(args.kwargs['sync_to_maisaka_history'])
        with self.assertRaises(ValueError):
            await self.bot._send(bad_frame(), 'chat', 'bad')
        self.bot.ctx.send.image.assert_awaited_once()

    async def test_short_raw_gif_base64_is_not_filtered_out(self):
        output = io.BytesIO()
        Image.new('RGB', (1, 1)).save(output, format='GIF')
        data = output.getvalue()
        encoded = base64.b64encode(data).decode()
        self.assertLess(len(encoded), 200)
        self.assertEqual(await self.bot._bytes_from_reference(encoded, ''), data)

    async def test_bad_gif_after_http_200_stays_unknown_without_second_post(self):
        for data in (bad_frame(), gif()[:-1]):
            post = self.post(fixtures.Response(json.dumps({'data': [{'b64_json': base64.b64encode(data).decode()}]}).encode()))
            self.bot._gemini = AsyncMock()
            self.bot.ctx.send = types.SimpleNamespace(image=AsyncMock())
            task = fixtures.mod.DrawTask('gif', 'chat', 'prompt', 'user')
            await self.bot._run_task(task)
            self.assertEqual(task.status, 'unknown')
            self.assertEqual(post.call_count, 1)
            self.bot._gemini.assert_not_awaited()
            self.bot.ctx.send.image.assert_not_awaited()
