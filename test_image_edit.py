import base64
import asyncio
import importlib.util
import io
import pathlib
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
from source_images import ImageCache, resolve_images, resolve_edit_images, resolve_context_image, extract, normalize_source_image

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC')


def message(mid='img'):
    return {'message_id': mid, 'message_segments': [
        {'type': 'image', 'binary_data_base64': base64.b64encode(PNG).decode()}]}


def sticker(data=PNG, mid='sticker'):
    return {'message_id': mid, 'raw_message': [
        {'type': 'emoji', 'data': '表情包文字描述不是图像', 'binary_data_base64': base64.b64encode(data).decode()}]}


def animation(fmt):
    output = io.BytesIO()
    first = Image.new('RGBA', (3, 2), 'red')
    second = Image.new('RGBA', (3, 2), 'blue')
    first.save(output, format=fmt, save_all=True, append_images=[second], duration=100, loop=0, lossless=True)
    return output.getvalue()


class Sources(unittest.IsolatedAsyncioTestCase):
    async def test_context_alias_and_tool_media_fallback_stay_in_current_chat(self):
        for reference in ['a7', 'tool-media-3']:
            async def fetch(cap, reference=reference, **kw):
                if cap == 'message.get_by_id':
                    return None
                self.assertEqual(cap, 'maisaka.context.resolve_image')
                self.assertEqual(kw, {'stream_id': 'chat-a', 'source_id': reference, 'index': 0})
                return {'success': True, 'image': {'binary_data_base64': base64.b64encode(PNG).decode()}}
            ctx = types.SimpleNamespace(call_capability=AsyncMock(side_effect=fetch))
            self.assertEqual(await resolve_edit_images(ctx, 'chat-a', reference, current=message('unrelated')), [PNG])

    async def test_explicit_context_index_never_queries_db_or_cache(self):
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={
            'success': True, 'image': {'binary_data_base64': base64.b64encode(PNG).decode()}}))
        cache = ImageCache()
        cache.put('chat', message('a7'))
        self.assertEqual(await resolve_edit_images(ctx, 'chat', 'a7', cache=cache, index=2), [PNG])
        ctx.call_capability.assert_awaited_once_with('maisaka.context.resolve_image', stream_id='chat', source_id='a7', index=2)

    async def test_bad_context_index_fails_without_any_lookup(self):
        ctx = types.SimpleNamespace(call_capability=AsyncMock())
        for index in [True, -1, '0', 1.5]:
            with self.assertRaises(ValueError):
                await resolve_edit_images(ctx, 'chat', 'img', current=message(), index=index)
        ctx.call_capability.assert_not_awaited()

    async def test_failed_selected_index_never_substitutes_cached_image(self):
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={'success': False, 'error': 'out of range'}))
        cache = ImageCache()
        cache.put('chat', message('img'))
        with self.assertRaises(ValueError):
            await resolve_edit_images(ctx, 'chat', 'img', cache=cache, index=7)
        self.assertEqual(ctx.call_capability.await_count, 1)

    async def test_legacy_sticker_and_quote_success_do_not_call_context_api(self):
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={'message': sticker()}))
        self.assertEqual(await resolve_edit_images(ctx, 'chat', 'sticker'), [PNG])
        self.assertEqual(ctx.call_capability.await_count, 1)
        self.assertEqual(ctx.call_capability.call_args.args, ('message.get_by_id',))

    async def test_decode_and_rpc_failures_do_not_trigger_alias_fallback(self):
        for value in [sticker(b'corrupt')]:
            ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={'message': value}))
            with self.assertRaises(ValueError):
                await resolve_edit_images(ctx, 'chat', 'img')
            self.assertEqual(ctx.call_capability.await_count, 1)
        ctx = types.SimpleNamespace(call_capability=AsyncMock(side_effect=RuntimeError('RPC too big')))
        with self.assertRaises(RuntimeError):
            await resolve_edit_images(ctx, 'chat', 'img')
        self.assertEqual(ctx.call_capability.await_count, 1)

    async def test_context_response_invalid_missing_and_oversized_bytes_rejected(self):
        for response in [None, {'success': False}, {'success': True},
                         {'success': True, 'image': {}},
                         {'success': True, 'image': {'binary_data_base64': 'bad'}},
                         {'success': True, 'image': {'binary_data_base64': base64.b64encode(b'corrupt').decode()}}]:
            ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value=response))
            with self.assertRaises(ValueError):
                await resolve_context_image(ctx, 'chat', 'media')
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={
            'success': True, 'image': {'binary_data_base64': base64.b64encode(PNG).decode()}}))
        with patch('source_images.MAX_BYTES', 10):
            with self.assertRaises(ValueError):
                await resolve_context_image(ctx, 'chat', 'media')
        with patch('source_images.MAX_PIXELS', 0):
            with self.assertRaises(ValueError):
                await resolve_context_image(ctx, 'chat', 'media')

    async def test_context_gif_is_preserved_and_cancellation_propagates(self):
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={
            'success': True, 'image': {'binary_data_base64': base64.b64encode(animation('GIF')).decode()}}))
        images = await resolve_context_image(ctx, 'chat', 'media')
        self.assertEqual(images, [animation('GIF')])
        with Image.open(io.BytesIO(images[0])) as image:
            self.assertEqual(image.format, 'GIF')
            self.assertEqual(image.convert('RGB').getpixel((0, 0)), (255, 0, 0))
        ctx.call_capability.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await resolve_context_image(ctx, 'chat', 'media')

    async def test_command_missing_source_does_not_guess_context_image(self):
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value=None))
        with self.assertRaises(ValueError):
            await resolve_edit_images(ctx, 'chat', current={'message_id': 'text-only'})
        self.assertTrue(all(call.args == ('message.get_by_id',) for call in ctx.call_capability.call_args_list))

    async def test_emoji_lookup_and_quote_are_chat_bound(self):
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={'message': sticker()}))
        images = await resolve_images(ctx, 'chat-a', current={'raw_message': [{'type': 'reply', 'data': {'target_message_id': 'sticker'}}]})
        self.assertEqual(images, [PNG])
        self.assertEqual(ctx.call_capability.call_args.kwargs, {'chat_id': 'chat-a', 'message_id': 'sticker', 'include_binary_data': True})

    async def test_emoji_cache_recovers_missing_binary_not_other_chats(self):
        cache = ImageCache()
        cache.put('chat-a', sticker())
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={'message': {'message_id': 'sticker', 'raw_message': [{'type': 'emoji', 'data': 'description'}]}}))
        self.assertEqual(await resolve_images(ctx, 'chat-a', 'sticker', cache=cache), [PNG])
        with self.assertRaises(ValueError):
            await resolve_images(ctx, 'chat-b', 'sticker', cache=cache)

    def test_static_sticker_is_preserved_and_deduplicated_with_image(self):
        m = message()
        m['message_segments'] += sticker()['raw_message']
        self.assertEqual(extract(m), [PNG])

    def test_webp_and_apng_sticker_use_first_static_frame(self):
        for fmt in ['WEBP', 'PNG']:
            data = animation(fmt)
            result = extract(sticker(data))[0]
            with Image.open(io.BytesIO(result)) as image:
                self.assertEqual(image.format, 'PNG')
                self.assertEqual(getattr(image, 'n_frames', 1), 1)
                self.assertEqual(image.convert('RGB').getpixel((0, 0)), (255, 0, 0))

    def test_gif_delivered_as_image_is_preserved(self):
        m = sticker(animation('GIF'))
        m['raw_message'][0]['type'] = 'image'
        result = extract(m)[0]
        self.assertEqual(result, animation('GIF'))
        with Image.open(io.BytesIO(result)) as image:
            self.assertEqual(image.format, 'GIF')
            self.assertEqual(image.convert('RGB').getpixel((0, 0)), (255, 0, 0))

    def test_apng_default_poster_is_not_first_animation_frame(self):
        output = io.BytesIO()
        poster = Image.new('RGBA', (3, 2), 'blue')
        frame = Image.new('RGBA', (3, 2), 'red')
        poster.save(output, format='PNG', save_all=True, append_images=[frame], default_image=True)
        result = extract(sticker(output.getvalue()))[0]
        with Image.open(io.BytesIO(result)) as image:
            self.assertEqual(image.convert('RGB').getpixel((0, 0)), (255, 0, 0))

    def test_truncated_jpeg_pixels_rejected_for_image_and_emoji(self):
        output = io.BytesIO()
        Image.new('RGB', (32, 32), 'red').save(output, format='JPEG')
        raw = output.getvalue()
        scan = raw.index(b'\xff\xda')
        header_end = scan + 2 + int.from_bytes(raw[scan+2:scan+4], 'big')
        truncated = raw[:header_end]
        for kind in ('image', 'emoji'):
            m = sticker(truncated)
            m['raw_message'][0]['type'] = kind
            with self.assertRaises(ValueError):
                extract(m)

    async def test_cache_is_usable_when_same_message_rpc_fails(self):
        cache = ImageCache()
        cache.put('a', sticker())
        ctx = types.SimpleNamespace(call_capability=AsyncMock(side_effect=RuntimeError('RPC unavailable')))
        self.assertEqual(await resolve_images(ctx, 'a', 'sticker', cache=cache), [PNG])
        with self.assertRaises(RuntimeError):
            await resolve_images(ctx, 'b', 'sticker', cache=cache)

    async def test_fresh_message_bytes_do_not_duplicate_stale_cache_version(self):
        cache = ImageCache()
        cache.put('a', sticker())
        output = io.BytesIO()
        Image.new('RGB', (2, 2), 'red').save(output, format='JPEG')
        fresh = output.getvalue()
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={'message': sticker(fresh)}))
        self.assertEqual(await resolve_images(ctx, 'a', 'sticker', cache=cache), [fresh])

    async def test_fresh_lookup_wins_over_current_message_old_bytes(self):
        output = io.BytesIO()
        Image.new('RGB', (2, 2), 'red').save(output, format='JPEG')
        fresh = output.getvalue()
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={'message': sticker(fresh)}))
        self.assertEqual(await resolve_images(ctx, 'a', current=sticker()), [fresh])

    async def test_cancelled_lookup_is_not_swallowed_by_cache(self):
        cache = ImageCache()
        cache.put('a', sticker())
        ctx = types.SimpleNamespace(call_capability=AsyncMock(side_effect=asyncio.CancelledError()))
        with self.assertRaises(asyncio.CancelledError):
            await resolve_images(ctx, 'a', 'sticker', cache=cache)

    async def test_quote_fanout_has_bounded_message_lookups(self):
        msg = {'message_id': 'root', 'raw_message': [{'type': 'reply', 'data': {'id': f'q-{i}'}} for i in range(17)]}
        async def fetch(cap, **kw):
            return {'message': msg if kw['message_id'] == 'root' else sticker(mid=kw['message_id'])}
        ctx = types.SimpleNamespace(call_capability=AsyncMock(side_effect=fetch))
        with self.assertRaisesRegex(ValueError, '16'):
            await resolve_images(ctx, 'chat', 'root')
        self.assertEqual(ctx.call_capability.call_count, 16)

    def test_description_unicode_and_qq_face_are_not_image_bytes(self):
        self.assertEqual(extract({'raw_message': [{'type': 'emoji', 'data': '描述'}, {'type': 'face', 'data': {'id': '123'}}, {'type': 'text', 'data': '😂'}]}), [])
        with self.assertRaises(ValueError):
            extract(sticker(b'not an image'))

    def test_sticker_limits_input_pixels_and_converted_output(self):
        with self.assertRaises(ValueError):
            normalize_source_image(b'x' * (10 * 1024 * 1024 + 1))
        with patch('source_images.MAX_PIXELS', 1):
            with self.assertRaises(ValueError):
                normalize_source_image(animation('GIF'))
        data = animation('PNG')
        with patch('source_images.MAX_BYTES', len(data)), patch.object(Image.Image, 'save', side_effect=lambda out, **kw: out.write(b'x' * (len(data) + 1))):
            with self.assertRaises(ValueError):
                normalize_source_image(data)

    def test_emoji_four_image_limit(self):
        raw = []
        for color in ['red', 'blue', 'green', 'yellow', 'black']:
            output = io.BytesIO()
            Image.new('RGB', (2, 2), color).save(output, format='PNG')
            raw.extend(sticker(output.getvalue())['raw_message'])
        with self.assertRaises(ValueError):
            extract({'raw_message': raw})

    async def test_quote_and_stream_scope(self):
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={'message': message()}))
        result = await resolve_images(ctx, 'chat-a', current={'message_segments': [{'type': 'reply', 'data': {'id': 'img'}}]})
        self.assertEqual(result, [PNG])
        self.assertEqual(ctx.call_capability.call_args.kwargs['chat_id'], 'chat-a')

    async def test_missing_explicit_image_never_uses_other_message(self):
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value=None))
        with self.assertRaises(ValueError):
            await resolve_images(ctx, 'chat-a', 'missing', current=message())

    async def test_cache_recovers_stripped_binary_and_isolates_chats(self):
        cache = ImageCache()
        cache.put('a', message())
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value=None))
        self.assertEqual(await resolve_images(ctx, 'a', 'img', cache=cache), [PNG])
        with self.assertRaises(ValueError):
            await resolve_images(ctx, 'b', 'img', cache=cache)
        cache.entries[('a', 'img')] = (0, [PNG])
        self.assertEqual(cache.get('a', 'img'), [])

    def test_duplicate_segments_and_bad_data(self):
        m = message()
        m['raw_message'] = m['message_segments']
        self.assertEqual(extract(m), [PNG])
        m['message_segments'][0]['binary_data_base64'] = 'not an image'
        with self.assertRaises(ValueError):
            extract(m)

    async def test_current_and_quoted_images_survive_cache_and_stripping(self):
        own = message('own')
        own['reply_to'] = 'quoted'
        quoted = message('quoted')
        other = PNG + b'distinct'
        quoted['message_segments'][0]['binary_data_base64'] = base64.b64encode(other).decode()
        stripped = {'message_id': 'own', 'reply_to': 'quoted', 'message_segments': [{'type': 'image'}]}
        for cached in (False, True):
            cache = ImageCache()
            if cached:
                cache.put('chat', own)
            async def get(cap, cached=cached, **kw):
                self.assertEqual(kw['chat_id'], 'chat')
                return {'message': (stripped if cached else own) if kw['message_id'] == 'own' else quoted}
            ctx = types.SimpleNamespace(call_capability=AsyncMock(side_effect=get))
            self.assertEqual(await resolve_images(ctx, 'chat', current=stripped, cache=cache), [PNG, other])
            self.assertEqual(await resolve_images(ctx, 'chat', 'own', cache=cache), [PNG, other])

    async def test_quote_cycle_is_bounded(self):
        m = message('self')
        m['reply_to'] = 'self'
        ctx = types.SimpleNamespace(call_capability=AsyncMock(return_value={'message': m}))
        self.assertEqual(await resolve_images(ctx, 'chat', 'self'), [PNG])
        self.assertEqual(ctx.call_capability.call_count, 1)


@unittest.skipUnless(importlib.util.find_spec('maibot_sdk'), 'requires MaiBot SDK (run on server)')
class Providers(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        package = types.ModuleType('neko_test_package')
        package.__path__ = [str(ROOT)]
        sys.modules[package.__name__] = package
        spec = importlib.util.spec_from_file_location('neko_test_package.plugin', ROOT / 'plugin.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self.mod = mod
        self.bot = object.__new__(mod.NekoDraw)

    def admission_fixture(self, bot):
        bot._generation_id = lambda: 0
        bot._current = lambda generation: True
        bot._reserve = lambda stream: True
        bot._reservations = {}
        return bot

    async def test_edit_tool_accepts_gif_sticker_without_generation(self):
        task = types.SimpleNamespace(summary=lambda: 'submitted')
        start = Mock(return_value=task)
        bot = types.SimpleNamespace(
            config=types.SimpleNamespace(plugin=types.SimpleNamespace(enabled=True)),
            ctx=types.SimpleNamespace(call_capability=AsyncMock(return_value={'message': sticker(animation('GIF'))})),
            _stream_id=lambda kw: 'chat', _active_task=lambda stream: None,
            _image_cache=ImageCache(), _start_task=start, _redact=str)
        result = await self.mod.NekoDraw.edit_image(self.admission_fixture(bot), prompt='换背景，保留人物', source_message_id='sticker', user_id='user')
        self.assertTrue(result['success'])
        args = start.call_args.args
        self.assertEqual(args[:3], ('chat', '换背景，保留人物\n高分辨率高帧', 'user'))
        with Image.open(io.BytesIO(args[3][0])) as image:
            self.assertEqual(image.format, 'GIF')
            self.assertEqual(image.convert('RGB').getpixel((0, 0)), (255, 0, 0))

        start.reset_mock()
        result = await self.mod.NekoDraw.edit_image(bot, prompt='高分辨率高帧，保留人物', source_message_id='sticker')
        self.assertTrue(result['success'])
        self.assertEqual(start.call_args.args[1], '高分辨率高帧，保留人物')

    async def test_edit_tool_submits_selected_context_image_without_generation(self):
        start = Mock(return_value=types.SimpleNamespace(summary=lambda: 'submitted'))
        cap = AsyncMock(return_value={'success': True, 'image': {'binary_data_base64': base64.b64encode(PNG).decode()}})
        bot = types.SimpleNamespace(config=types.SimpleNamespace(plugin=types.SimpleNamespace(enabled=True)),
                                    ctx=types.SimpleNamespace(call_capability=cap), _stream_id=lambda kw: 'chat',
                                    _active_task=lambda stream: None, _image_cache=ImageCache(), _start_task=start, _redact=str)
        result = await self.mod.NekoDraw.edit_image(self.admission_fixture(bot), prompt='换背景', source_message_id='tool-media-2', source_image_index=1)
        self.assertTrue(result['success'])
        self.assertEqual(start.call_args.args[3], [PNG])
        cap.assert_awaited_once_with('maisaka.context.resolve_image', stream_id='chat', source_id='tool-media-2', index=1)

    async def test_chat_keeps_image_bytes(self):
        self.bot._post_json = AsyncMock(return_value={})
        self.bot._images_from_openai_body = AsyncMock(return_value=[PNG])
        cfg = self.mod.OpenAIConfig(api_key='test')
        await self.bot._openai_chat('change background', cfg, [PNG])
        content = self.bot._post_json.call_args.args[1]['messages'][0]['content']
        self.assertEqual(content[0]['text'], 'change background')
        self.assertEqual(content[1]['image_url']['url'], 'data:image/png;base64,' + base64.b64encode(PNG).decode())

    async def test_edits_multipart_is_recreated_on_each_attempt(self):
        class Response:
            status = 200
            @property
            def content(self): return self
            async def iter_chunked(self, size): yield b'{}'
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def text(self): return '{}'
            async def json(self, **kwargs): return {}
        forms = []
        def post(url, **kwargs):
            self.assertTrue(url.endswith('/v1/images/edits'))
            forms.append(kwargs['data'])
            self.assertNotIn('Content-Type', kwargs['headers'])
            return Response()
        self.bot._http = AsyncMock(return_value=types.SimpleNamespace(post=post))
        self.bot._images_from_openai_body = AsyncMock(return_value=[PNG])
        for _ in range(2):
            await self.bot._openai_images('edit', self.mod.OpenAIConfig(api_key='test'), [PNG])
        self.assertIsNot(forms[0], forms[1])
        self.assertEqual(forms[0]._fields[-1][0]['name'], 'image[]')
        self.assertEqual(forms[0]._fields[-1][2], PNG)

    async def test_gemini_keeps_inline_image(self):
        encoded = base64.b64encode(PNG).decode()
        post = AsyncMock(return_value={'candidates': [{'content': {'parts': [{'inlineData': {'data': encoded}}]}}]})
        bot = types.SimpleNamespace(config=types.SimpleNamespace(gemini=self.mod.GeminiConfig(api_key='test')),
                                    _post_json=post)
        bot._gemini_result = self.bot._gemini_result
        result = await self.mod.NekoDraw._gemini_native(bot, 'edit background', [PNG])
        self.assertEqual(result, PNG)
        parts = post.call_args.args[1]['contents'][0]['parts']
        self.assertEqual(parts[1]['inlineData'], {'mimeType': 'image/png', 'data': encoded})

    async def test_generation_and_edit_route_preferences_are_independent(self):
        async def retry(task, label, fn):
            return await fn()
        images = AsyncMock(return_value=PNG)
        chat = AsyncMock(return_value=PNG)
        bot = types.SimpleNamespace(config=types.SimpleNamespace(openai=self.mod.OpenAIConfig(api_key='test', mode='auto')),
                                    _preferred_openai_mode='chat', _preferred_openai_edit_mode='',
                                    _openai_images=images, _openai_chat=chat, _call_with_retry=retry)
        task = types.SimpleNamespace(images=[PNG])
        await self.mod.NekoDraw._openai(bot, 'edit', task)
        images.assert_awaited_once()
        chat.assert_not_awaited()
        self.assertEqual(bot._preferred_openai_mode, 'chat')
        self.assertEqual(bot._preferred_openai_edit_mode, 'images')
        await self.mod.NekoDraw._openai(bot, 'generate')
        chat.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
