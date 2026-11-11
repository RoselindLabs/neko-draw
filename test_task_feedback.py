import asyncio
import importlib.util
import json
import pathlib
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('maibot_sdk'), 'requires MaiBot SDK')
class TaskFeedback(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = pathlib.Path(__file__).parent
        pkg = types.ModuleType('draw_feedback_test'); pkg.__path__ = [str(root)]; sys.modules[pkg.__name__] = pkg
        spec = importlib.util.spec_from_file_location('draw_feedback_test.plugin', root/'plugin.py')
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        self.mod = mod
        class Bot(mod.NekoDraw):
            config = mod.PluginConfig()
            ctx = types.SimpleNamespace(logger=Mock())
        self.bot = object.__new__(Bot)
        self.bot._tasks = {}; self.bot._latest_by_stream = {}; self.bot._runners = {}
        self.task = mod.DrawTask('1234abcd', 'chat-a', 'test prompt', 'user')
        self.bot._remember_task(self.task)

    async def test_current_and_legacy_snapshots_replace_old_queue_state(self):
        for key in ('items', 'messages'):
            self.task.touch('queued')
            payload = {'session_id':'chat-a', key:[]}
            payload = (await self.bot.configure_planner(**payload))['modified_kwargs']
            self.assertIn('尚未开始', json.dumps(payload, ensure_ascii=False))
            self.task.touch('succeeded')
            payload = (await self.bot.configure_planner(**payload))['modified_kwargs']
            self.assertIn('已完成', json.dumps(payload, ensure_ascii=False))
            payload = (await self.bot.configure_planner(**payload))['modified_kwargs']
            self.assertEqual(payload[key], [])
            self.assertIsNone(await self.bot.configure_planner(**payload))

    async def test_failure_remains_visible_on_subsequent_planner_calls(self):
        self.task.touch('failed', error=self.bot._policy_feedback('', 'HTTP 400：{"error":{"message":"openai_error"}}'))
        result = await self.bot.configure_planner(session_id='chat-a', items=[])
        text = json.dumps(result, ensure_ascii=False)
        self.assertIn('已失败或取消', text)
        self.assertNotIn('有时是提示词被拦截', text)
        self.assertIn('原因不明', self.bot._status_text('chat-a'))
        self.assertIsNone(await self.bot.configure_planner(session_id='chat-a', items=[]))
        self.assertIsNone(await self.bot.configure_planner(session_id='chat-b', items=[]))

    async def test_send_confirmation_controls_terminal_status(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def send(*args):
            entered.set()
            await release.wait()
        self.bot._generate = AsyncMock(return_value=(b'image', 'openai'))
        self.bot._send = send
        self.task.touch('running')
        runner = asyncio.create_task(self.bot._run_task(self.task))
        await entered.wait()
        self.assertEqual(self.task.status, 'running')
        release.set()
        await runner
        self.assertEqual(self.task.status, 'succeeded')
        self.bot._send = AsyncMock(side_effect=RuntimeError('delivery failed'))
        self.task.touch('running')
        await self.bot._run_task(self.task)
        self.assertEqual(self.task.status, 'unknown')
        self.assertIn('不自动补发', self.task.summary())

    async def test_uncertain_transport_never_retries_or_falls_back(self):
        import aiohttp

        self.bot.config.general.retry_times = 3
        self.bot.config.general.fallback_provider = 'gemini'
        self.bot.config.openai.mode = 'auto'
        self.bot._preferred_openai_mode = ''
        for error in [asyncio.TimeoutError(), aiohttp.ServerDisconnectedError(), self.mod.GenerationUncertain('HTTP 502')]:
            self.bot._openai_images = AsyncMock(side_effect=error)
            self.bot._openai_chat = AsyncMock()
            self.bot._gemini = AsyncMock()
            with self.assertRaises(self.mod.GenerationUncertain):
                await self.bot._generate('test', self.task)
            self.bot._openai_images.assert_awaited_once()
            self.bot._openai_chat.assert_not_awaited()
            self.bot._gemini.assert_not_awaited()

    async def test_explicit_rate_limit_still_retries_with_redacted_log(self):
        self.bot.config.general.retry_times = 1
        self.bot.config.openai.api_key = 'private-test-key'
        call = AsyncMock(side_effect=[self.mod.GenerationNotSubmitted('HTTP 429：private-test-key'), b'image'])
        with patch.object(self.mod.asyncio, 'sleep', AsyncMock()):
            self.assertEqual(await self.bot._call_with_retry(self.task, 'test', call), b'image')
        self.assertEqual(call.await_count, 2)
        self.assertNotIn('private-test-key', str(self.bot.ctx.logger.info.call_args_list))

    async def test_http_timeout_and_server_errors_are_uncertain_on_both_post_routes(self):
        for status in [408, 500, 502, 503, 504]:
            response = types.SimpleNamespace(status=status)
            context = AsyncMock()
            context.__aenter__.return_value = response
            self.bot._http = AsyncMock(return_value=types.SimpleNamespace(post=Mock(return_value=context)))
            with patch.object(self.mod, 'read_response', AsyncMock(return_value=b'upstream error')):
                with self.assertRaises(self.mod.GenerationUncertain):
                    await self.bot._post_json('https://example.com', {}, {}, '')
                self.bot.config.openai.api_key = 'synthetic'
                with patch.object(self.mod, 'mime_type', return_value='image/png'):
                    with self.assertRaises(self.mod.GenerationUncertain):
                        await self.bot._openai_images('edit', self.bot.config.openai, [b'synthetic'])

    async def test_unknown_generation_and_send_cancellation_preserve_unknown(self):
        self.bot._generate = AsyncMock(side_effect=self.mod.GenerationUncertain('supplier unknown'))
        self.bot._send = AsyncMock()
        self.task.images = [b'secret']
        await self.bot._run_task(self.task)
        self.assertEqual(self.task.status, 'unknown')
        self.assertEqual(self.task.images, [])
        self.bot._send.assert_not_awaited()
        response = await self.bot.configure_planner(session_id='chat-a', items=[])
        self.assertIn('不要声称成功或确定失败', str(response))
        self.bot._generate = AsyncMock(return_value=(b'image', 'fake'))
        self.bot._send = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await self.bot._run_task(self.task)
        self.assertEqual(self.task.status, 'unknown')

    async def test_internal_reference_instructions_not_used_for_history_or_summary(self):
        self.task.prompt = 'internal reference rules'
        self.task.display_prompt = '窗边阅读'
        self.bot._generate = AsyncMock(return_value=(b'image', 'fake'))
        self.bot._send = AsyncMock()
        await self.bot._run_task(self.task)
        self.bot._generate.assert_awaited_once_with('internal reference rules', self.task)
        self.bot._send.assert_awaited_once_with(b'image', 'chat-a', '窗边阅读')
        self.assertNotIn('internal reference rules', self.task.summary())

    async def test_disabled_command_and_planner_do_not_start_or_advertise_drawing(self):
        self.bot.config.plugin.enabled = False
        self.bot.config.command.allowed_user_ids = ['123']
        self.bot._start_task = Mock()
        from test_tool_context import command_context
        result = await self.bot.cmd_draw(**command_context('chat-a'), matched_groups={'payload':'cat'})
        self.assertFalse(result[0])
        self.bot._start_task.assert_not_called()
        self.assertIsNone(await self.bot.configure_planner(session_id='chat-a', messages=[{'role':'user','content':'帮我画一张图'}]))

    async def test_lost_state_after_reload_does_not_repeat_old_receipt(self):
        self.bot._tasks.clear(); self.bot._latest_by_stream.clear()
        result = await self.bot.configure_planner(session_id='chat-a', messages=[
            {'role':'tool', 'content':'neko_edit_image 提交成功；状态：排队'}])
        self.assertIsNone(result)
        self.assertIn('不能据历史提交回执', self.bot._status_text('chat-a'))

    async def test_expiry_dual_payload_and_new_task(self):
        self.task.touch('failed', error='private details')
        with patch.object(self.mod.time, 'monotonic', return_value=self.task.terminal_at + 900):
            self.assertIsNone(await self.bot.configure_planner(session_id='chat-a', items=[], messages=[]))
        self.assertFalse(self.task.terminal_hint_used)
        result = await self.bot.configure_planner(session_id='chat-a', items=[], messages=[])
        self.assertEqual(len(result['modified_kwargs']['items']), 1)
        self.assertEqual(len(result['modified_kwargs']['messages']), 1)
        self.assertNotIn('private details', json.dumps(result))
        cleaned = await self.bot.configure_planner(**result['modified_kwargs'])
        self.assertEqual(cleaned['modified_kwargs']['items'], [])
        self.assertEqual(cleaned['modified_kwargs']['messages'], [])
        new = self.mod.DrawTask('new-task', 'chat-a', 'new prompt', 'user')
        self.bot._remember_task(new)
        new.touch('succeeded')
        self.assertIsNotNone(await self.bot.configure_planner(session_id='chat-a', items=[]))

    async def test_cleanup_preserves_user_tool_and_other_system(self):
        marker = '[neko.draw 当前任务状态]'
        self.bot._tasks.clear()
        keep = [{'item_type': 'UserMessageItem', 'parts': [{'text': marker}]},
                {'item_type': 'ToolOutputItem', 'content': '状态：排队'}, self.bot._system_item('other '+marker)]
        result = await self.bot.configure_planner(session_id='chat-a', items=[self.bot._system_item(marker+' old'), *keep])
        self.assertEqual(result['modified_kwargs']['items'], keep)

    async def test_no_payload_does_not_consume_terminal_hint(self):
        self.task.touch('succeeded')
        self.assertIsNone(await self.bot.configure_planner(session_id='chat-a'))
        self.assertFalse(self.task.terminal_hint_used)

    def test_eviction_cleans_latest_reference(self):
        self.task.touch('failed')
        for n in range(31):
            task = self.mod.DrawTask(str(n), 'other-'+str(n), 'prompt', 'user')
            task.touch('succeeded')
            self.bot._remember_task(task)
        self.assertNotIn('chat-a', self.bot._latest_by_stream)

    def test_error_evidence_is_preserved_without_inventing_cause(self):
        for raw in ('HTTP 400：{"error":{"message":"openai_error"}}', 'HTTP 400：broken JSON',
                    'HTTP 400：{"error":{"code":"openai_error"}}'):
            text = self.bot._policy_feedback('prompt', raw)
            self.assertIn('原因不明', text)
            self.assertIn('不能仅凭 HTTP 400 断言源图 ID 错误', text)
            self.assertIn('有时是提示词被拦截', text)
            self.assertIn('修改提示词后重新生成', text)
            self.assertNotIn('只试一次', text)
        specific = self.bot._policy_feedback('', 'HTTP 400：{"error":{"message":"Invalid size: 999x999"}}')
        self.assertIn('Invalid size: 999x999', specific)
        self.assertNotIn('具体原因未知', specific)
