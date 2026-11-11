"""Real SDK schema/context -> tool -> queued task -> JSON/multipart, offline only."""
import asyncio
import base64
import copy
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image
from maibot_sdk.context import PluginContext
from maibot_sdk.types import ToolComponentInfo
import test_admission_safety as fixtures
from test_admission_safety import Response, mod
from test_tool_context import tool_context, SOURCE_PNG


def image_bytes(size, fmt='PNG'):
    out = io.BytesIO()
    image = Image.new('RGB', size, 'red')
    if fmt == 'GIF':
        image.save(out, format=fmt, save_all=True,
                   append_images=[Image.new('RGB', size, 'blue')], duration=100, loop=0)
    else:
        image.save(out, format=fmt)
    return out.getvalue()


class AspectRatioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixtures.Safety.asyncSetUp(self)
        self.bot.config.general.fallback_provider = 'none'
        self.bot.config.openai.mode = 'images'
        self.sources = [image_bytes((160, 90))]
        self.rpc_calls = []
        self.posts = []
        self.response_override = None

        async def rpc(method, plugin_id, payload, **kwargs):
            self.assertEqual(method, 'cap.call')
            self.rpc_calls.append(payload)
            if payload['capability'] == 'message.get_by_id':
                self.assertNotEqual(payload['args']['chat_id'], 'forged')
                return {'success': True, 'message': {'message_id': 'm', 'message_segments': [
                    {'type': 'image', 'binary_data_base64': base64.b64encode(data).decode()}
                    for data in self.sources]}}
            if payload['capability'] == 'send.image':
                return {'success': True}
            if payload['capability'] == 'maisaka.context.resolve_image':
                self.assertEqual(payload['args']['index'], 1)
                return {'success': True, 'image': {'binary_data_base64': base64.b64encode(self.sources[1]).decode()}}
            raise AssertionError(payload['capability'])

        self.bot.ctx = PluginContext('neko.draw', rpc_call=rpc)

        def post(url, **kwargs):
            self.posts.append((url, kwargs))
            if self.response_override is not None:
                return self.response_override
            encoded = base64.b64encode(SOURCE_PNG).decode()
            body = {'candidates': [{'content': {'parts': [{'inlineData': {'data': encoded}}]}}]} if 'generateContent' in url else {'data': [{'b64_json': 'data:image/png;base64,' + encoded}]}
            return Response(json.dumps(body).encode())

        self.bot._http = AsyncMock(return_value=types.SimpleNamespace(post=post))

    def definition(self, name):
        component = next(item for item in self.bot.get_components() if item['name'] == name)
        info = ToolComponentInfo(name=name, **component['metadata'])
        return component, {'type': 'function', 'function': {
            'name': name, 'description': info.get_brief_description(),
            'parameters': info.get_parameters_schema()}}

    async def invoke(self, name='neko_draw', **args):
        component, _ = self.definition(name)
        result = await getattr(self.bot, component['metadata']['handler_name'])(
            prompt='test composition', _host_context=tool_context(), stream_id='forged', **args)
        runners = list(self.bot._runners.values())
        if runners:
            await asyncio.gather(*runners)
        return result, self.bot._tasks.get(self.bot._latest_by_stream.get('chat'))

    async def test_real_sdk_schema_dynamic_descriptions_and_native_explicit_ratios(self):
        self.bot.config.general.provider = 'gemini'
        self.bot.config.gemini.model = 'gemini-3.1-flash-image'
        definitions = [self.definition(name)[1] for name in ('neko_draw', 'neko_edit_image')]
        for enabled in (False, True):
            self.bot.config.self_portrait.enabled = enabled
            result = await self.bot.configure_planner(session_id='chat', items=[], tool_definitions=copy.deepcopy(definitions))
            for definition in result['modified_kwargs']['tool_definitions']:
                function = definition['function']
                schema = function['parameters']
                self.assertNotIn('aspect_ratio', schema['required'])
                self.assertIn('16:9', schema['properties']['aspect_ratio']['enum'])
                self.assertIn('用户指定', function['description'])
                self.assertIn('chat', schema['properties']['aspect_ratio']['description'])
            self.assertIn('主动选择', result['modified_kwargs']['tool_definitions'][0]['function']['description'])
        for ratio in ('16:9', '9:16'):
            result, task = await self.invoke(aspect_ratio=ratio)
            self.assertTrue(result['success'])
            self.assertEqual(task.status, 'succeeded')
            self.assertEqual(task.aspect_ratio, ratio)
            self.assertEqual(self.posts[-1][1]['json']['generationConfig']['imageConfig'], {'aspectRatio': ratio})
            self.assertIn(ratio, (await self.bot.draw_status(_host_context=tool_context()))['content'])
            self.assertIn('aspectRatio', task.summary())

    async def test_omitted_draw_and_reference_keep_original_configuration(self):
        self.bot.config.openai.size = '1024x1536'
        _, task = await self.invoke()
        self.assertIsNone(task.aspect_ratio)
        self.assertEqual(self.posts[-1][1]['json']['size'], '1024x1536')
        self.bot.config.general.provider = 'gemini'
        self.bot.config.gemini.aspect_ratio = '4:3'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ref.png'
            path.write_bytes(image_bytes((20, 80)))
            self.bot.config.self_portrait.enabled = True
            self.bot.config.self_portrait.reference_path = str(path)
            result, task = await self.invoke(use_self_reference=True)
        self.assertTrue(result['success'])
        self.assertIsNone(task.aspect_ratio)
        self.assertEqual(self.posts[-1][1]['json']['generationConfig']['imageConfig'], {'aspectRatio': '4:3'})
        self.assertEqual(self.bot.config.openai.size, '1024x1536')

    async def test_gpt_image2_json_and_real_multipart_sizes(self):
        for ratio, size in (('16:9', '1536x864'), ('9:16', '864x1536'), ('4:3', '1024x768')):
            _, task = await self.invoke(aspect_ratio=ratio)
            self.assertEqual(task.status, 'succeeded')
            self.assertEqual(self.posts[-1][1]['json']['size'], size)
            _, task = await self.invoke('neko_edit_image', source_message_id='m', aspect_ratio=ratio)
            self.assertEqual(task.status, 'succeeded')
            self.assertTrue(self.posts[-1][0].endswith('/images/edits'))
            form = self.posts[-1][1]['data']
            writer = types.SimpleNamespace(write=AsyncMock())
            await form().write(writer)
            wire = b''.join(call.args[0] for call in writer.write.await_args_list)
            self.assertIn(b'name="size"\r\n\r\n' + size.encode(), wire)
            self.assertIn(self.sources[0], wire)
        self.assertEqual(self.bot.config.openai.size, '1024x1024')

    async def test_legacy_gpt_images_approximation_is_exposed(self):
        for model in ('gpt-image-1', 'gpt-image-1-mini', 'gpt-image-1.5'):
            self.bot.config.openai.model = model
            _, task = await self.invoke(aspect_ratio='16:9')
            self.assertEqual(self.posts[-1][1]['json']['size'], '1536x1024')
            self.assertIn('近似为3:2，非精确16:9', task.summary())
        # Infer against actual source, not a previous nearest-ratio rounding.
        self.sources = [image_bytes((122, 100))]
        _, task = await self.invoke('neko_edit_image', source_message_id='m')
        fields = {field[0]['name']: field[2] for field in self.posts[-1][1]['data']._fields}
        self.assertEqual(fields['size'], '1024x1024')

    async def test_chat_modes_have_only_prompt_guidance_no_fabricated_fields(self):
        for provider in ('openai', 'gemini'):
            self.bot.config.general.provider = provider
            self.bot.config.openai.mode = 'chat'
            self.bot.config.gemini.mode = 'openai'
            for name, extra in (('neko_draw', {}), ('neko_edit_image', {'source_message_id': 'm'})):
                _, task = await self.invoke(name, aspect_ratio='9:16', **extra)
                payload = self.posts[-1][1]['json']
                self.assertEqual(set(payload), {'model', 'messages', 'stream'})
                self.assertIn('9:16', str(payload['messages']))
                self.assertIn('不保证输出尺寸', task.summary())
                self.assertEqual(task.status, 'succeeded')

    async def test_edit_source_first_canvas_reference_and_explicit_precedence(self):
        self.bot.config.general.provider = 'gemini'
        self.sources = [image_bytes((160, 90), 'GIF'), image_bytes((60, 90))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ref.png'
            reference = image_bytes((30, 90))
            path.write_bytes(reference)
            self.bot.config.self_portrait.enabled = True
            self.bot.config.self_portrait.reference_path = str(path)
            for value, expected in ((None, '16:9'), ('source', '16:9'), ('9:16', '9:16')):
                _, task = await self.invoke('neko_edit_image', source_message_id='m', use_self_reference=True, aspect_ratio=value)
                payload = self.posts[-1][1]['json']
                self.assertEqual(payload['generationConfig']['imageConfig']['aspectRatio'], expected)
                parts = payload['contents'][0]['parts']
                self.assertEqual([base64.b64decode(part['inlineData']['data']) for part in parts[1:]], [*self.sources, reference])
                self.assertEqual(parts[1]['inlineData']['mimeType'], 'image/gif')
                self.assertEqual(task.source_canvas, (160, 90) if value != '9:16' else None)
                if value != '9:16':
                    self.assertIn('160×90', task.summary())

    def test_animation_canvas_header_does_not_decode_frames(self):
        raw = image_bytes((160, 90), 'GIF')
        with patch.object(Image.Image, 'load', side_effect=AssertionError('no additional pixel decode')):
            self.assertEqual(mod.source_canvas(raw), (160, 90))

    async def test_selected_context_index_is_primary_canvas(self):
        self.bot.config.general.provider = 'gemini'
        self.sources = [image_bytes((160, 90)), image_bytes((90, 160))]
        _, task = await self.invoke('neko_edit_image', source_message_id='tool-media', source_image_index=1)
        self.assertEqual(task.status, 'succeeded')
        self.assertEqual(task.source_canvas, (90, 160))
        self.assertEqual(task.aspect_ratio, '9:16')
        self.assertEqual(self.posts[-1][1]['json']['generationConfig']['imageConfig']['aspectRatio'], '9:16')
        self.assertEqual([call['capability'] for call in self.rpc_calls], ['maisaka.context.resolve_image', 'send.image'])

    async def test_invalid_values_fail_before_sources_references_tasks_or_http(self):
        for value in ('', 'auto', '21:9', '9999x9999', '1:0', 1, True, [], {}):
            for name in ('neko_draw', 'neko_edit_image'):
                result, task = await self.invoke(name, aspect_ratio=value)
                self.assertFalse(result['success'])
                self.assertIsNone(task)
        result, _ = await self.invoke(aspect_ratio='source')
        self.assertFalse(result['success'])
        self.assertEqual(self.posts, [])
        self.assertEqual(self.rpc_calls, [])
        self.assertFalse(self.bot._reservations)

    async def test_unregistered_models_fail_locally_without_fallback(self):
        self.bot.config.general.fallback_provider = 'gemini'
        self.bot.config.openai.model = 'unverified-gpt-image-alias'
        _, task = await self.invoke(aspect_ratio='16:9')
        self.assertEqual(task.status, 'failed')
        self.assertIn('未登记', task.error)
        self.bot.config.general.provider = 'gemini'
        self.bot.config.general.fallback_provider = 'openai'
        self.bot.config.gemini.model = 'unverified-image-model'
        _, task = await self.invoke(aspect_ratio='16:9')
        self.assertEqual(task.status, 'failed')
        self.assertEqual(self.posts, [])

    async def test_ratio_survives_queue_and_cannot_bypass_same_chat_admission(self):
        self.bot._generation_slots = asyncio.Semaphore(0)
        result = await self.bot.draw(prompt='scene', aspect_ratio='16:9', _host_context=tool_context())
        self.assertTrue(result['success'])
        task = self.bot._tasks[self.bot._latest_by_stream['chat']]
        self.assertEqual(task.status, 'queued')
        self.assertIn('16:9', result['content'])
        again = await self.bot.draw(prompt='scene', aspect_ratio='9:16', _host_context=tool_context())
        self.assertFalse(again['success'])
        self.assertEqual(len(self.bot._runners), 1)
        self.bot._generation_slots.release()
        await asyncio.gather(*list(self.bot._runners.values()))
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0][1]['json']['size'], '1536x864')
        hooked = await self.bot.configure_planner(session_id='chat', items=[])
        self.assertIn('1536x864', str(hooked))

    async def test_unknown_still_one_post_no_fallback_or_send(self):
        self.bot.config.general.retry_times = 3
        self.bot.config.general.fallback_provider = 'gemini'
        self.bot.config.openai.mode = 'auto'
        self.response_override = Response(b'{invalid')
        _, task = await self.invoke(aspect_ratio='16:9')
        self.assertEqual(task.status, 'unknown')
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.rpc_calls, [])
        self.assertIn('1536x864', task.summary())

    async def test_auto_route_fallback_retains_ratio_and_announces_chat_limit(self):
        self.bot.config.openai.mode = 'auto'
        original = self.bot._http.return_value.post

        def post(url, **kwargs):
            if '/images/' in url:
                self.posts.append((url, kwargs))
                return Response(status=404)
            return original(url, **kwargs)

        self.bot._http.return_value.post = post
        _, task = await self.invoke(aspect_ratio='16:9')
        self.assertEqual(len(self.posts), 2)
        self.assertEqual(task.status, 'succeeded')
        self.assertIn('16:9', str(self.posts[1][1]['json']))
        self.assertIn('chat仅', task.summary())

    async def test_life_remains_disabled_and_does_not_inherit_ratio(self):
        self.assertFalse(self.bot.config.life.enabled)
        result = await self.bot.life_v1({'aspect_ratio': '16:9'})
        self.assertEqual(result['status'], 'disabled')
        self.assertFalse(self.bot._tasks)
        self.assertFalse(self.posts)

    async def test_unknown_alias_omissions_preserve_native_and_images_requests(self):
        for provider in ('openai', 'gemini'):
            self.bot.config.general.provider = provider
            cfg = getattr(self.bot.config, provider)
            cfg.model = 'private-image-alias'
            for name, extra in (('neko_draw', {}), ('neko_edit_image', {'source_message_id': 'm'})):
                result, task = await self.invoke(name, **extra)
                self.assertTrue(result['success'])
                self.assertEqual(task.status, 'succeeded')
                payload = self.posts[-1][1]
                if provider == 'gemini':
                    self.assertEqual(payload['json']['generationConfig']['imageConfig'], {'aspectRatio': '1:1'})
                elif name == 'neko_draw':
                    self.assertEqual(payload['json']['size'], '1024x1024')
                else:
                    fields = {field[0]['name']: field[2] for field in payload['data']._fields}
                    self.assertEqual(fields['size'], '1024x1024')
                if name == 'neko_edit_image':
                    self.assertIn('源比例未应用', task.summary())
                    self.assertIn('源比例未应用', result['content'])

    async def test_unknown_alias_explicit_source_and_ratio_never_post(self):
        for provider in ('openai', 'gemini'):
            self.bot.config.general.provider = provider
            getattr(self.bot.config, provider).model = 'private-image-alias'
            for name, ratio in (('neko_draw', '16:9'), ('neko_edit_image', '16:9'), ('neko_edit_image', 'source')):
                _, task = await self.invoke(name, aspect_ratio=ratio, **({'source_message_id': 'm'} if name == 'neko_edit_image' else {}))
                self.assertEqual(task.status, 'failed')
                self.assertIn('显式aspect_ratio/source未提交', task.error)
        self.assertFalse(self.posts)

    async def test_unknown_auto_cold_and_cached_have_same_capability_policy(self):
        self.bot.config.openai.model = 'private-image-alias'
        self.bot.config.openai.mode = 'auto'
        for preferred in ('', 'chat'):
            for name, extra in (('neko_draw', {}), ('neko_edit_image', {'source_message_id': 'm'})):
                self.bot._preferred_openai_mode = preferred
                self.bot._preferred_openai_edit_mode = preferred
                before = len(self.posts)
                _, task = await self.invoke(name, aspect_ratio='16:9', **extra)
                self.assertEqual(task.status, 'failed')
                self.assertEqual(len(self.posts), before)
                # Omission retains the original successful-route preference;
                # both cold and cached routes work, neither receives a ratio.
                _, task = await self.invoke(name, **extra)
                self.assertEqual(task.status, 'succeeded')
                self.assertEqual(len(self.posts), before + 1)
                if preferred == 'chat':
                    self.assertTrue(self.posts[-1][0].endswith('/chat/completions'))
                    self.assertNotIn('构图建议', str(self.posts[-1][1]['json']))
                else:
                    self.assertIn('/images/', self.posts[-1][0])
                if name == 'neko_edit_image':
                    self.assertIn('源比例未应用', task.summary())

    async def test_unknown_backup_does_not_veto_known_preferred_success(self):
        for provider, fallback in (('openai', 'gemini'), ('gemini', 'openai')):
            self.bot.config.general.provider = provider
            self.bot.config.general.fallback_provider = fallback
            getattr(self.bot.config, provider).model = 'gpt-image-2' if provider == 'openai' else 'gemini-3.1-flash-image'
            getattr(self.bot.config, fallback).model = 'private-image-alias'
            before = len(self.posts)
            _, task = await self.invoke(aspect_ratio='16:9')
            self.assertEqual(task.status, 'succeeded')
            self.assertEqual(len(self.posts), before + 1)
            self.assertEqual(task.provider, provider)

    async def test_unknown_backup_is_checked_only_when_reached(self):
        self.bot.config.general.provider = 'gemini'
        self.bot.config.general.fallback_provider = 'openai'
        self.bot.config.openai.model = 'private-image-alias'
        original = self.bot._http.return_value.post

        def post(url, **kwargs):
            if 'generateContent' in url:
                self.posts.append((url, kwargs))
                return Response(status=404)
            return original(url, **kwargs)

        self.bot._http.return_value.post = post
        _, task = await self.invoke('neko_edit_image', source_message_id='m')
        self.assertEqual(task.status, 'succeeded')
        self.assertEqual(len(self.posts), 2)
        fields = {field[0]['name']: field[2] for field in self.posts[-1][1]['data']._fields}
        self.assertEqual(fields['size'], '1024x1024')
        self.assertIn('源比例未应用', task.summary())
        before = len(self.posts)
        _, task = await self.invoke('neko_edit_image', source_message_id='m', aspect_ratio='source')
        self.assertEqual(task.status, 'failed')
        self.assertEqual(len(self.posts), before + 1)  # Only the known route's explicit 404.
        self.assertIn('显式aspect_ratio/source未提交', task.error)

    async def test_unknown_route_planner_hints_do_not_induce_rejected_arguments(self):
        for provider in ('openai', 'gemini'):
            self.bot.config.general.provider = provider
            getattr(self.bot.config, provider).model = 'private-image-alias'
            for mode in (('images', 'auto') if provider == 'openai' else ('native',)):
                getattr(self.bot.config, provider).mode = mode
                for preferred in ('', 'chat'):
                    self.bot._preferred_openai_mode = preferred
                    definitions = [self.definition(name)[1] for name in ('neko_draw', 'neko_edit_image')]
                    result = await self.bot.configure_planner(session_id='chat', messages=[{'role': 'user', 'content': '请画一张猫'}], tool_definitions=definitions)
                    for definition in result['modified_kwargs']['tool_definitions']:
                        function = definition['function']
                        self.assertIn('不要主动添加aspect_ratio', function['description'])
                        self.assertIn('不要主动添加aspect_ratio', function['parameters']['properties']['aspect_ratio']['description'])
                    self.assertIn('不要主动添加aspect_ratio', result['modified_kwargs']['messages'][-1]['content'])
                    self.assertNotIn('按画面主动选择', result['modified_kwargs']['messages'][-1]['content'])


if __name__ == '__main__':
    unittest.main()
