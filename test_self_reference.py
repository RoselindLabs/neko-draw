import asyncio
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image
from self_reference import read_reference, MAX_REFERENCE_BYTES, portrait_prompt
from test_tool_context import tool_context


def image_bytes(fmt='JPEG'):
    buf = io.BytesIO()
    Image.new('RGB', (24, 32), 'blue').save(buf, format=fmt)
    return buf.getvalue()


class ReferenceTests(unittest.TestCase):
    def test_static_formats_preserve_exact_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            for fmt in ['PNG', 'JPEG', 'WEBP']:
                path=Path(directory)/('ref.'+fmt.lower());data=image_bytes(fmt);path.write_bytes(data)
                self.assertEqual(read_reference(str(path)), data)

    def test_directory_missing_relative_corrupt_and_oversize_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ref.jpg'
            for value in [directory, str(path), 'relative.jpg']:
                with self.assertRaises((ValueError,OSError)):read_reference(value)
            for data in [b'',b'not-an-image',b'x'*(MAX_REFERENCE_BYTES+1)]:
                path.write_bytes(data)
                with self.assertRaises((ValueError,OSError)):read_reference(str(path))

    def test_animation_and_excessive_pixels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ref.png'
            first=Image.new('RGB',(8,8),'blue');second=Image.new('RGB',(8,8),'red')
            first.save(path,format='PNG',save_all=True,append_images=[second],duration=100)
            with self.assertRaises(ValueError):read_reference(str(path))
            path.write_bytes(image_bytes())
            with patch('self_reference.MAX_REFERENCE_PIXELS',100):
                with self.assertRaises(ValueError):read_reference(str(path))

    def test_symlink_is_rejected_before_open_without_platform_privilege(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ref.jpg';path.write_bytes(image_bytes())
            with patch.object(Path,'is_symlink',return_value=True),patch('self_reference.os.open',side_effect=AssertionError('must not read')):
                with self.assertRaises(ValueError):read_reference(str(path))

    def test_replaced_inode_or_changed_stat_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ref.jpg';path.write_bytes(image_bytes())
            with patch('self_reference.os.path.samestat',return_value=False):
                with self.assertRaises(ValueError):read_reference(str(path))


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        pkg=types.ModuleType('draw_reference_test');pkg.__path__=[str(Path(__file__).parent)]
        sys.modules[pkg.__name__]=pkg
        self.module=importlib.import_module(pkg.__name__+'.plugin')
        module=self.module
        class Bot(module.NekoDraw):
            config=module.PluginConfig()
            ctx=types.SimpleNamespace(logger=Mock())
        self.bot=Bot();self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.addAsyncCleanup(self.bot.on_unload)
        self.path=Path(self.temp.name)/'ref.jpg';self.data=image_bytes();self.path.write_bytes(self.data)
        self.bot.config.self_portrait.enabled=True;self.bot.config.self_portrait.reference_path=str(self.path)
        self.bot._start_task=Mock(return_value=types.SimpleNamespace(summary=lambda:'queued'))

    async def test_explicit_reference_attaches_original_and_uses_edit_path(self):
        result=await self.bot.draw(prompt='在窗边读书的自画像',use_self_reference=True,_host_context=tool_context())
        self.assertTrue(result['success'])
        self.assertIn('已附角色参考图',result['content'])
        args=self.bot._start_task.call_args.args
        self.assertEqual(args[3],[self.data]);self.assertIn('视觉依据',args[1]);self.assertIn('窗边',args[1])
        self.assertNotIn(str(self.path),args[1])

    async def test_normal_drawing_never_reads_reference_or_attaches_it(self):
        with patch.object(self.module,'read_reference',side_effect=AssertionError('ordinary drawing must not read')):
            result=await self.bot.draw(prompt='无人环境',_host_context=tool_context())
        self.assertTrue(result['success']);self.assertEqual(len(self.bot._start_task.call_args.args),3)

    async def test_missing_or_disabled_reference_denies_without_text_fallback(self):
        for disabled in [False,True]:
            self.bot.config.self_portrait.enabled=not disabled
            self.bot.config.self_portrait.reference_path=str(self.path)+'-missing'
            result=await self.bot.draw(prompt='self portrait',use_self_reference=True,_host_context=tool_context())
            self.assertFalse(result['success'])
        self.bot._start_task.assert_not_called()

    async def test_reconfiguration_during_file_read_denies(self):
        original=asyncio.to_thread
        async def change(*args):
            value=await original(*args)
            self.bot.config.self_portrait.enabled=False
            return value
        with patch.object(self.module.asyncio,'to_thread',side_effect=change):
            result=await self.bot.draw(prompt='self portrait',use_self_reference=True,_host_context=tool_context())
        self.assertFalse(result['success']);self.bot._start_task.assert_not_called()

    async def test_racing_active_task_after_file_read_does_not_start(self):
        self.bot._active_task=Mock(side_effect=[None,types.SimpleNamespace(summary=lambda:'running')])
        result=await self.bot.draw(prompt='self portrait',use_self_reference=True,_host_context=tool_context())
        self.assertFalse(result['success']);self.bot._start_task.assert_not_called()

    async def test_invalid_bool_and_cancellation_do_not_generate(self):
        for value in ['true',1,None]:
            self.assertFalse((await self.bot.draw('self',use_self_reference=value,_host_context=tool_context()))['success'])
        async def cancel(*args):raise asyncio.CancelledError
        with patch.object(self.module.asyncio,'to_thread',side_effect=cancel):
            with self.assertRaises(asyncio.CancelledError):
                await self.bot.draw('self',use_self_reference=True,_host_context=tool_context())
        self.bot._start_task.assert_not_called()

    async def test_planner_prefers_reference_without_path_or_memory_prerequisite(self):
        text=self.bot._planner_instruction(True)
        self.assertIn('use_self_reference=true',text);self.assertIn('不必先调用query_memory',text)
        self.assertNotIn(str(self.path),text)
        result=await self.bot.configure_planner(stream_id='chat',items=[{'item_type':'UserMessageItem','parts':[{'type':'text','text':'画你自己在读书'}]}])
        self.assertIn('use_self_reference=true',json.dumps(result,ensure_ascii=False))

    async def test_gemini_native_receives_reference_as_inline_image(self):
        self.bot.config.gemini.api_key='fake-key'
        self.bot._post_json=AsyncMock(return_value={'candidates':[{'content':{'parts':[{'inlineData':{'data':__import__('base64').b64encode(image_bytes('PNG')).decode()}}]}}]})
        await self.bot._gemini_native(portrait_prompt('读书'),[self.data])
        payload=self.bot._post_json.call_args.args[1]
        parts=payload['contents'][0]['parts']
        self.assertEqual(parts[1]['inlineData']['mimeType'],'image/jpeg')
        self.assertEqual(__import__('base64').b64decode(parts[1]['inlineData']['data']),self.data)

    async def test_edit_appends_reference_and_preserves_source_order(self):
        sources=[image_bytes('PNG'),image_bytes('WEBP')]
        with patch.object(self.module,'resolve_edit_images',AsyncMock(return_value=sources)) as resolve:
            result=await self.bot.edit_image('把我换成机器人角色，保留背景','source',use_self_reference=True,_host_context=tool_context())
        self.assertTrue(result['success'])
        args=self.bot._start_task.call_args.args
        self.assertEqual(args[3],[*sources,self.data]);self.assertEqual(sources,[image_bytes('PNG'),image_bytes('WEBP')])
        self.assertIn('前2张',args[1]);self.assertIn('最后1张',args[1]);self.assertNotIn(str(self.path),args[1])
        self.assertIn('2 张聊天源图',result['content']);self.assertIn('附1张',result['content'])
        self.assertEqual(resolve.call_args.args[1:3],('chat','source'))

    async def test_ordinary_edit_does_not_read_or_append_reference(self):
        with patch.object(self.module,'resolve_edit_images',AsyncMock(return_value=[image_bytes('PNG')])),patch.object(self.module,'read_reference',side_effect=AssertionError('must not read')):
            result=await self.bot.edit_image('仅改背景','source',_host_context=tool_context())
        self.assertTrue(result['success']);self.assertEqual(self.bot._start_task.call_args.args[1],'仅改背景')
        self.assertEqual(self.bot._start_task.call_args.args[3],[image_bytes('PNG')])

    async def test_edit_requires_real_source_even_with_reference(self):
        for value in [[],ValueError('missing source')]:
            resolver=AsyncMock(side_effect=value) if isinstance(value,Exception) else AsyncMock(return_value=value)
            with patch.object(self.module,'resolve_edit_images',resolver),patch.object(self.module,'read_reference',side_effect=AssertionError('no source no reference')):
                result=await self.bot.edit_image('edit','source',use_self_reference=True,_host_context=tool_context())
            self.assertFalse(result['success'])
        self.bot._start_task.assert_not_called()

    async def test_edit_four_sources_rejected_without_dropping_any(self):
        sources=[image_bytes('PNG')]*4
        with patch.object(self.module,'resolve_edit_images',AsyncMock(return_value=sources)),patch.object(self.module,'read_reference',side_effect=AssertionError('over limit')):
            result=await self.bot.edit_image('edit','source',use_self_reference=True,_host_context=tool_context())
        self.assertFalse(result['success']);self.assertIn('source_image_index',result['content']);self.assertEqual(len(sources),4)
        self.bot._start_task.assert_not_called()

    async def test_edit_three_sources_plus_reference_is_four(self):
        with patch.object(self.module,'resolve_edit_images',AsyncMock(return_value=[image_bytes('PNG')]*3)):
            result=await self.bot.edit_image('edit','source',use_self_reference=True,_host_context=tool_context())
        self.assertTrue(result['success']);self.assertEqual(len(self.bot._start_task.call_args.args[3]),4)

    async def test_edit_invalid_or_disabled_reference_has_no_fallback(self):
        with patch.object(self.module,'resolve_edit_images',AsyncMock(return_value=[image_bytes('PNG')])):
            self.bot.config.self_portrait.reference_path+='-missing'
            self.assertFalse((await self.bot.edit_image('edit','source',use_self_reference=True,_host_context=tool_context()))['success'])
            self.bot.config.self_portrait.enabled=False
            self.assertFalse((await self.bot.edit_image('edit','source',use_self_reference=True,_host_context=tool_context()))['success'])
        self.bot._start_task.assert_not_called()

    async def test_edit_reference_reconfigured_or_plugin_disabled_during_read(self):
        original=asyncio.to_thread
        for option in ['reference','plugin']:
            self.bot.config.self_portrait.enabled=True;self.bot.config.plugin.enabled=True
            async def change(*args):
                result=await original(*args)
                if option=='reference':self.bot.config.self_portrait.enabled=False
                else:self.bot.config.plugin.enabled=False
                return result
            with patch.object(self.module,'resolve_edit_images',AsyncMock(return_value=[image_bytes('PNG')])),patch.object(self.module.asyncio,'to_thread',side_effect=change):
                self.assertFalse((await self.bot.edit_image('edit','source',use_self_reference=True,_host_context=tool_context()))['success'])
        self.bot._start_task.assert_not_called()

    async def test_edit_cancellation_and_invalid_parameters_never_start(self):
        with patch.object(self.module,'resolve_edit_images',AsyncMock()) as resolve:
            for value in ['true',1,None]:
                self.assertFalse((await self.bot.edit_image('edit','source',use_self_reference=value,_host_context=tool_context()))['success'])
            resolve.assert_not_called()
        async def cancel(*args):raise asyncio.CancelledError
        with patch.object(self.module,'resolve_edit_images',AsyncMock(return_value=[image_bytes('PNG')])),patch.object(self.module.asyncio,'to_thread',side_effect=cancel):
            with self.assertRaises(asyncio.CancelledError):await self.bot.edit_image('edit','source',use_self_reference=True,_host_context=tool_context())
        self.bot._start_task.assert_not_called()

    @staticmethod
    def _tool_definitions():
        def entry(name, flag_text='flag'):
            properties = {'use_self_reference': {'type': 'boolean', 'description': flag_text}} if flag_text else {}
            return {'type': 'function', 'function': {'name': name, 'description': 'registered', 'parameters': {'type': 'object', 'properties': properties}}}
        return [entry('reply'), entry('neko_draw'), entry('neko_edit_image'), entry('neko_draw_status', None)]

    async def test_tool_hints_follow_self_reference_switch(self):
        definitions = self._tool_definitions()
        result = await self.bot.configure_planner(session_id='chat', items=[], tool_definitions=definitions)
        payload = result['modified_kwargs']['tool_definitions']
        by_name = {tool['function']['name']: tool['function'] for tool in payload}
        self.assertIn('无需先查query_memory', by_name['neko_draw']['description'])
        self.assertIn('use_self_reference=true', by_name['neko_draw']['description'])
        self.assertIn('角色参考附最后', by_name['neko_edit_image']['parameters']['properties']['use_self_reference']['description'])
        self.assertIn('设true，插件附管理员角色参考图', by_name['neko_draw']['parameters']['properties']['use_self_reference']['description'])
        self.assertEqual(by_name['reply']['description'], 'registered')
        self.assertEqual(by_name['neko_draw_status']['description'], 'registered')
        self.assertNotIn('query_memory', by_name['neko_edit_image']['parameters']['properties']['use_self_reference']['description'])

    async def test_tool_hints_without_reference_return_to_memory_guidance(self):
        self.bot.config.self_portrait.enabled = False
        definitions = self._tool_definitions()
        result = await self.bot.configure_planner(session_id='chat', items=[], tool_definitions=definitions)
        by_name = {tool['function']['name']: tool['function'] for tool in result['modified_kwargs']['tool_definitions']}
        self.assertIn('先用query_memory查长期记忆外貌', by_name['neko_draw']['description'])
        self.assertIn('未配置角色参考图', by_name['neko_draw']['parameters']['properties']['use_self_reference']['description'])
        self.assertIn('保持false', by_name['neko_edit_image']['parameters']['properties']['use_self_reference']['description'])
        self.assertNotIn('use_self_reference=true', by_name['neko_draw']['description'])
        self.assertEqual(by_name['reply']['description'], 'registered')

    async def test_disabled_plugin_and_foreign_definitions_are_left_alone(self):
        definitions = self._tool_definitions()
        result = await self.bot.configure_planner(session_id='chat', items=[], tool_definitions=definitions)
        self.assertEqual([tool['function']['description'] for tool in result['modified_kwargs']['tool_definitions']
                          if tool['function']['name'] not in ('neko_draw', 'neko_edit_image')], ['registered', 'registered'])
        self.bot.config.plugin.enabled = False
        self.assertIsNone(await self.bot.configure_planner(session_id='chat', items=[], tool_definitions=self._tool_definitions()))
