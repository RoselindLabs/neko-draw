"""Offline regression barriers: no supplier, QQ, host or external network calls."""
import asyncio
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch
import aiohttp

from test_life_authorization import packet
from test_tool_context import tool_context


ROOT = Path(__file__).parent
pkg = types.ModuleType('draw_safety')
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
spec = importlib.util.spec_from_file_location(pkg.__name__ + '.plugin', ROOT / 'plugin.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class Response:
    def __init__(self, raw=b'{}', status=200, error=None):
        self.raw, self.status, self.error = raw, status, error
        self.content = self

    async def iter_chunked(self, size):
        if self.error:
            raise self.error
        yield self.raw

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Safety(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        class Bot(mod.NekoDraw):
            config = mod.PluginConfig()
            ctx = types.SimpleNamespace(logger=Mock())
        self.bot = Bot()
        self.bot.config.openai.api_key = 'synthetic-secret-value'
        self.bot.config.gemini.enabled = True
        self.bot.config.gemini.api_key = 'synthetic-other-secret'
        self.bot.config.general.retry_times = 0
        self.addAsyncCleanup(self.bot.on_unload)

    def post(self, response):
        post = Mock(return_value=response)
        self.bot._http = AsyncMock(return_value=types.SimpleNamespace(post=post))
        return post

    async def test_200_failure_matrix_never_posts_twice_or_falls_back(self):
        for editing in (False, True):
            for case, response in (
                ('invalid-json', Response(b'not-json')),
                ('non-object', Response(b'[]')),
                ('no-image', Response()),
                ('corrupt-image', Response(json.dumps({'data': [{'b64_json': 'A' * 204}]}).encode())),
                ('read-over-limit', Response(error=ValueError('response too large'))),
                ('url-safety', Response(b'{"data":[{"url":"https://synthetic.invalid/result"}]}')),
                ('url-timeout', Response(b'{"data":[{"url":"https://synthetic.invalid/result"}]}')),
            ):
                with self.subTest(editing=editing, case=case):
                    post = self.post(response)
                    self.bot._download = AsyncMock(side_effect=TimeoutError() if case == 'url-timeout' else ValueError('unsafe URL'))
                    self.bot._gemini = AsyncMock()
                    task = mod.DrawTask('test', 'chat', 'prompt', 'user')
                    task.images = [b'\x89PNG\r\n\x1a\nsource'] if editing else []
                    await self.bot._run_task(task)
                    self.assertEqual(task.status, 'unknown')
                    self.assertEqual(post.call_count, 1)
                    self.bot._gemini.assert_not_awaited()

    async def test_gemini_200_invalid_results_do_not_fallback(self):
        self.bot.config.general.provider = 'gemini'
        self.bot.config.general.fallback_provider = 'openai'
        self.bot._openai = AsyncMock()
        for raw in (b'{}', b'{"candidates":[{"content":"invalid"}]}',
                    b'{"candidates":[{"content":{"parts":[{"inlineData":{"data":"invalid"}}]}}]}'):
            post = self.post(Response(raw))
            with self.assertRaises(mod.GenerationUncertain):
                await self.bot._generate('prompt')
            self.assertEqual(post.call_count, 1)
        self.bot._openai.assert_not_awaited()

    async def test_explicit_rejection_can_fallback_but_arbitrary_exception_cannot(self):
        post = self.post(Response(status=404))
        self.bot._gemini = AsyncMock(return_value=b'image')
        self.assertEqual(await self.bot._generate('prompt'), (b'image', 'gemini'))
        self.assertEqual(post.call_count, 2)  # two explicitly rejected OpenAI routes
        self.bot._gemini.reset_mock()
        self.bot.config.general.retry_times = 2
        self.bot._openai_images = AsyncMock(side_effect=RuntimeError('unexpected HTTP 429 in arbitrary text'))
        with self.assertRaises(RuntimeError):
            await self.bot._generate('prompt')
        self.bot._openai_images.assert_awaited_once()
        self.bot._gemini.assert_not_awaited()

    async def test_connector_establishment_and_real_429_keep_retry_policy(self):
        self.bot.config.openai.mode = 'images'
        self.bot.config.general.retry_times = 1
        self.bot._gemini = AsyncMock(return_value=b'image')
        key = types.SimpleNamespace(host='synthetic.invalid', port=443, ssl=True)
        connector = aiohttp.ClientConnectorError(key, OSError('synthetic connection refused'))
        for response in (Response(status=429), connector):
            with self.subTest(response=type(response).__name__):
                post = self.post(response)
                if isinstance(response, Exception):
                    post.side_effect = response
                with patch.object(mod.asyncio, 'sleep', new_callable=AsyncMock) as sleep:
                    self.assertEqual(await self.bot._generate('prompt'), (b'image', 'gemini'))
                self.assertEqual(post.call_count, 2)
                sleep.assert_awaited_once()

    async def test_edit_and_portrait_old_generation_cannot_revive_after_reload(self):
        for entry in ('edit', 'portrait', 'edit-portrait'):
            with self.subTest(entry=entry):
                entered, release = asyncio.Event(), asyncio.Event()
                async def barrier(*args, entered=entered, release=release, entry=entry, **kwargs):
                    entered.set()
                    await release.wait()
                    return [b'source'] if entry == 'edit' else b'portrait'
                self.bot.config.self_portrait.enabled = True
                self.bot.config.self_portrait.reference_path = 'synthetic-only-not-opened'
                self.bot._generate = AsyncMock()
                self.bot._send = AsyncMock()
                with patch.object(mod, 'resolve_edit_images', side_effect=barrier if entry == 'edit' else AsyncMock(return_value=[b'source'])), patch.object(mod, 'image_work', side_effect=barrier):
                    call = (self.bot.draw(prompt='prompt', use_self_reference=True, _host_context=tool_context())
                            if entry == 'portrait' else self.bot.edit_image(prompt='prompt', _host_context=tool_context(), use_self_reference=entry == 'edit-portrait'))
                    pending = asyncio.create_task(call)
                    await entered.wait()
                    await self.bot.on_config_update('', {}, 'test')
                    release.set()
                    self.assertFalse((await pending)['success'])
                self.assertFalse(self.bot._runners)
                self.bot._generate.assert_not_awaited()
                self.bot._send.assert_not_awaited()

    async def test_unload_during_send_validation_never_calls_qq(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def barrier(*args):
            entered.set()
            await release.wait()
        self.bot.ctx.send = types.SimpleNamespace(image=AsyncMock())
        with patch.object(mod, 'image_work', side_effect=barrier):
            pending = asyncio.create_task(self.bot._send(b'image', 'chat', 'prompt'))
            await entered.wait()
            await self.bot.on_config_update('', {}, 'test')
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        self.bot.ctx.send.image.assert_not_awaited()

    async def test_life_busy_never_consumes_and_reserves_against_regular_draw(self):
        with tempfile.TemporaryDirectory() as directory:
            self.bot.ctx.paths = types.SimpleNamespace(data_dir=directory)
            Path(directory, 'life-keys.json').write_text(json.dumps({'neko.life': 'ab' * 32}), encoding='utf-8')
            Path(directory, 'life-keys.json').chmod(0o600)
            self.bot.config.life.enabled = True
            self.bot.config.life.sessions = ['chat']
            body = {'session_id': 'chat', 'request_id': '12' * 16, 'prompt': 'scene'}
            self.assertTrue(self.bot._reserve('chat'))
            value = packet(body)
            self.assertEqual((await self.bot.life_v1(packet=value))['status'], 'busy')
            with closing(sqlite3.connect(Path(directory, 'life-admission.db'))) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0], 0)
                self.assertEqual(db.execute('SELECT count(*) FROM nonces').fetchone()[0], 0)
            self.bot._reservations.pop('chat')
            entered, release = asyncio.Event(), asyncio.Event()
            original = mod.consume_life
            async def barrier(func, *args, **kwargs):
                entered.set()
                await release.wait()
                return original(*args, **kwargs)
            self.bot._generate = AsyncMock(return_value=(b'image', 'fake'))
            self.bot._send = AsyncMock()
            with patch.object(mod.asyncio, 'to_thread', side_effect=barrier):
                pending = asyncio.create_task(self.bot.life_v1(packet=value))
                await entered.wait()
                self.assertFalse((await self.bot.draw(prompt='ordinary', _host_context=tool_context()))['success'])
                release.set()
                self.assertEqual((await pending)['status'], 'submitted')
            await asyncio.gather(*self.bot._runners.values())
            self.bot._tasks.clear()
            self.assertTrue(self.bot._reserve('chat'))
            self.assertEqual((await self.bot.life_v1(packet=packet(body)))['status'], 'unknown')
            self.bot._generate.assert_awaited_once()

    async def test_bounded_capacity_generation_queue_cancel_and_prompt(self):
        entered, release = asyncio.Event(), asyncio.Event()
        active = peak = 0
        async def generate(*args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            entered.set()
            try:
                await release.wait()
                return b'image', 'fake'
            finally:
                active -= 1
        self.bot._generate = generate
        self.bot._send = AsyncMock()
        self.assertFalse((await self.bot.draw(prompt='x' * 4001, _host_context=tool_context('long')))['success'])
        results = await asyncio.gather(*(self.bot.draw(prompt='scene', _host_context=tool_context(f'chat-{i}')) for i in range(mod.MAX_ADMITTED + 3)))
        self.assertEqual(sum(r['success'] for r in results), mod.MAX_ADMITTED)
        await entered.wait()
        await asyncio.sleep(0)
        self.assertEqual(peak, mod.MAX_GENERATING)
        await self.bot.on_unload()
        self.assertFalse(self.bot._runners)
        self.assertFalse(self.bot._reservations)
        self.bot._send.assert_not_awaited()

    async def test_preprocessing_reserves_global_capacity_before_any_runner(self):
        entered = asyncio.Event()
        count = 0
        async def barrier(*args, **kwargs):
            nonlocal count
            count += 1
            if count == mod.MAX_ADMITTED:
                entered.set()
            await asyncio.Event().wait()
        with patch.object(mod, 'resolve_edit_images', side_effect=barrier) as resolver:
            tasks = [asyncio.create_task(self.bot.edit_image(prompt='edit', _host_context=tool_context(f'chat-{i}')))
                     for i in range(mod.MAX_ADMITTED)]
            try:
                await asyncio.wait_for(entered.wait(), 2)
                self.assertFalse(self.bot._runners)
                self.assertFalse((await self.bot.draw(prompt='scene', _host_context=tool_context('extra')))['success'])
                self.assertFalse((await self.bot.edit_image(prompt='edit', _host_context=tool_context('chat-0')))['success'])
                self.assertEqual(resolver.await_count, mod.MAX_ADMITTED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.assertFalse(self.bot._reservations)

    async def test_life_ingress_waiters_are_bounded_and_cancel_releases_budget(self):
        await self.bot._life_lock.acquire()
        tasks = [asyncio.create_task(self.bot.life_v1(packet={})) for _ in range(mod.MAX_ADMITTED)]
        try:
            await asyncio.sleep(0)
            self.assertEqual(self.bot._life_pending, mod.MAX_ADMITTED)
            self.assertEqual((await self.bot.life_v1(packet={}))['status'], 'busy')
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.bot._life_lock.release()
        self.assertEqual(self.bot._life_pending, 0)

    async def test_error_key_straddling_truncation_never_leaks(self):
        key = self.bot.config.openai.api_key
        for status in (400, 429):
            raw = ('x' * 490 + key + ' suffix').encode()
            self.post(Response(raw, status))
            with self.assertRaises((mod.PromptRejected, mod.GenerationNotSubmitted)) as caught:
                await self.bot._post_json('synthetic', {}, {}, '')
            self.assertNotIn(key[:10], str(caught.exception))
            self.assertIn('***', str(caught.exception))

    async def test_preprocessing_cancellation_propagates_and_releases_admission(self):
        entered = asyncio.Event()
        async def barrier(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        with patch.object(mod, 'resolve_edit_images', side_effect=barrier):
            pending = asyncio.create_task(self.bot.edit_image(prompt='edit', _host_context=tool_context()))
            await entered.wait()
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        self.assertFalse(self.bot._reservations)


class WorkerBudget(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_threads_keep_both_slots_until_completion(self):
        from resource_budget import image_work
        loop = asyncio.get_running_loop()
        entered, third = asyncio.Event(), asyncio.Event()
        stop = threading.Event()
        count = 0
        def work():
            nonlocal count
            count += 1
            if count == 2:
                loop.call_soon_threadsafe(entered.set)
            if count == 3:
                loop.call_soon_threadsafe(third.set)
            stop.wait(5)
        tasks = [asyncio.create_task(image_work(work)) for _ in range(3)]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            tasks[0].cancel()
            await asyncio.sleep(0)
            tasks[0].cancel()
            await asyncio.sleep(0)
            self.assertFalse(third.is_set())
            self.assertFalse(tasks[0].done())
        finally:
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertTrue(third.is_set())
