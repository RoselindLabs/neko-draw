"""Independent-review regressions: decoded secrets, stale queues, real threads."""
import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import AsyncMock, patch

import test_admission_safety as fixtures
from test_life_authorization import packet
from test_tool_context import tool_context

mod = fixtures.mod


class ReviewSafety(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.Safety.asyncSetUp
    post = fixtures.Safety.post

    def life(self, directory):
        self.bot.ctx.paths = types.SimpleNamespace(data_dir=directory)
        path = Path(directory, 'life-keys.json')
        path.write_text(json.dumps({'neko.life': 'ab' * 32}), encoding='utf-8')
        path.chmod(0o600)
        self.bot.config.life.enabled = True
        self.bot.config.life.sessions = ['chat']
        return {'session_id': 'chat', 'request_id': '56' * 16, 'prompt': 'scene'}

    async def test_unicode_secret_decoded_before_every_error_exit_and_truncation(self):
        for field in ('message', 'code', 'type', 'error-string', 'top-message'):
            for secret in (self.bot.config.openai.api_key, self.bot.config.gemini.api_key):
                for status in (400, 429):
                    with self.subTest(field=field, secret=secret, status=status):
                        text = 'x' * 290 + secret + ' tail'
                        body = ({'error': text} if field == 'error-string' else
                                {'message': text} if field == 'top-message' else {'error': {field: text}})
                        raw = json.dumps(body).replace(secret[0] + secret[1:], r'\u0073' + secret[1:])
                        self.assertNotIn(secret, raw)
                        self.assertNotIn(secret, self.bot._api_error_detail(raw))
                        direct = self.bot._policy_feedback('prompt', 'HTTP 400：' + raw)
                        self.assertNotIn(secret, direct)
                        self.assertNotIn(secret[:10], direct)  # redact before 300-char detail cut
                        self.post(fixtures.Response(raw.encode(), status))
                        with self.assertRaises((mod.PromptRejected, mod.GenerationNotSubmitted)) as caught:
                            await self.bot._post_json('synthetic', {}, {}, '')
                        self.assertNotIn(secret, str(caught.exception))
                        self.assertNotIn(secret[:10], str(caught.exception))
                        task = mod.DrawTask('redaction', 'chat', 'prompt', 'user')
                        self.bot._remember_task(task)
                        self.bot._generate = AsyncMock(side_effect=caught.exception)
                        await self.bot._run_task(task)
                        outlets = [task.error, task.summary(), self.bot._status_text('chat'),
                                   str(self.bot.ctx.logger.mock_calls)]
                        self.assertTrue(all(secret not in value for value in outlets))

    async def test_overlapping_keys_redact_longest_first(self):
        self.bot.config.gemini.api_key = self.bot.config.openai.api_key + '-longer'
        self.assertEqual(self.bot._redact(self.bot.config.gemini.api_key), '***')

    async def test_json_error_unrecognized_fields_never_reach_outlets(self):
        for status in (400, 429):
            self.post(fixtures.Response(br'{"debug":"\u0073ynthetic-secret-value"}', status))
            with self.assertRaises((mod.PromptRejected, mod.GenerationNotSubmitted)) as caught:
                await self.bot._post_json('synthetic', {}, {}, '')
            self.assertNotIn('ynthetic-secret', str(caught.exception))
            self.assertNotIn('debug', str(caught.exception))

    async def test_queued_deadline_expires_without_slot_release_or_post(self):
        self.bot._generate = AsyncMock()
        self.bot._send = AsyncMock()
        slots = self.bot._generation_slots
        for _ in range(mod.MAX_GENERATING):
            await slots.acquire()
        try:
            task = self.bot._start_task('chat', 'prompt', 'user', [b'source'])
            task.admission_deadline = time.monotonic() + 0.02
            await asyncio.wait_for(self.bot._runners[task.task_id], 1)
            self.assertEqual(task.status, 'failed')
            self.assertIn('未提交供应商', task.error)
            self.assertFalse(task.images)
            self.assertIsNone(self.bot._active_task('chat'))
            self.bot._generate.assert_not_awaited()
            self.bot._send.assert_not_awaited()
            # Expiry must not manufacture or leak a permit.
            self.assertTrue(slots.locked())
        finally:
            for _ in range(mod.MAX_GENERATING):
                slots.release()

    async def test_age_is_rechecked_after_slot_acquisition(self):
        task = mod.DrawTask('expired', 'chat', 'prompt', 'user')
        task.admission_deadline = time.monotonic() + 60
        self.bot._generate = AsyncMock()
        slots = self.bot._generation_slots
        original = slots.acquire
        now = time.monotonic()
        async def acquire():
            nonlocal now
            result = await original()
            # Simulate scheduling delay at the boundary without a wall-clock wait.
            now += 120
            return result
        # Only the plugin's clock is replaced; asyncio keeps its real clock.
        fake_time = types.SimpleNamespace(monotonic=lambda: now, time=time.time)
        with patch.object(mod, 'time', fake_time), patch.object(slots, 'acquire', side_effect=acquire):
            await self.bot._run_task(task)
        self.assertEqual(task.status, 'failed')
        self.bot._generate.assert_not_awaited()
        self.assertEqual(slots._value, mod.MAX_GENERATING)

    async def test_preprocessing_deadline_is_not_reset_when_task_is_created(self):
        self.bot._generate = AsyncMock()
        async def source(*args, **kwargs):
            generation, _ = self.bot._reservations['chat']
            self.bot._reservations['chat'] = (generation, time.monotonic() - 1)
            return [b'source']
        with patch.object(mod, 'resolve_edit_images', side_effect=source):
            result = await self.bot.edit_image(prompt='edit', _host_context=tool_context())
        self.assertFalse(result['success'])
        self.assertIn('未提交供应商', result['content'])
        self.assertFalse(self.bot._reservations)
        self.assertFalse(self.bot._runners)
        self.bot._generate.assert_not_awaited()

    async def test_deadline_does_not_reclassify_posted_result_failure_or_cancel(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def response_body(*args):
            entered.set()
            await release.wait()
            return b'not-json'
        post = self.post(fixtures.Response())
        self.bot.config.openai.mode = 'images'
        with patch.object(mod, 'read_response', side_effect=response_body):
            task = self.bot._start_task('chat', 'prompt', 'user')
            runner = self.bot._runners[task.task_id]
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(task.status, 'running')
            task.admission_deadline = time.monotonic() - 1
            release.set()
            await runner
        self.assertEqual(task.status, 'unknown')
        self.assertEqual(post.call_count, 1)
        entered.clear()
        release.clear()
        with patch.object(mod, 'read_response', side_effect=response_body):
            task = self.bot._start_task('chat-cancel', 'prompt', 'user')
            runner = self.bot._runners[task.task_id]
            await asyncio.wait_for(entered.wait(), 1)
            task.admission_deadline = time.monotonic() - 1
            runner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await runner
        self.assertEqual(task.status, 'unknown')
        self.assertEqual(post.call_count, 2)

    async def test_life_lock_wait_deadline_never_releases_another_owners_lock(self):
        lock = asyncio.Lock()
        await lock.acquire()
        try:
            with self.assertRaises(TimeoutError):
                async with mod.deadline_slot(lock, time.monotonic() + 0.02):
                    self.fail('expired waiter acquired lock')
            self.assertTrue(lock.locked())
        finally:
            lock.release()

    async def test_life_expiry_after_consume_is_failed_not_unknown_and_never_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            body = self.life(directory)
            now = time.monotonic()
            fake_time = types.SimpleNamespace(monotonic=lambda: now, time=time.time)
            original = mod.consume_life
            def consume(*args, **kwargs):
                nonlocal now
                result = original(*args, **kwargs)
                now += self.bot.config.general.timeout + 1
                return result
            self.bot._generate = AsyncMock()
            with patch.object(mod, 'time', fake_time), patch.object(mod, 'consume_life', side_effect=consume):
                result = await self.bot.life_v1(packet=packet(body))
            self.assertEqual(result['status'], 'expired')
            task = self.bot._tasks[body['request_id']]
            self.assertEqual(task.status, 'failed')
            self.assertIn('未提交供应商', task.error)
            self.assertEqual((await self.bot.life_v1(packet=packet(body)))['status'], 'failed')
            self.assertFalse(self.bot._runners)
            self.assertFalse(self.bot._reservations)
            self.bot._generate.assert_not_awaited()

    async def test_life_queue_rechecks_policy_and_current_host_target(self):
        with tempfile.TemporaryDirectory() as directory:
            body = self.life(directory)
            for revoke in ('policy', 'host'):
                with self.subTest(revoke=revoke):
                    self.bot.config.life.enabled = True
                    self.bot.config.life.sessions = ['chat']
                    self.bot._generate = AsyncMock()
                    self.bot._send = AsyncMock()
                    await self.bot._generation_slots.acquire()
                    await self.bot._generation_slots.acquire()
                    try:
                        request = {**body, 'request_id': ('78' if revoke == 'policy' else '90') * 16}
                        result = await self.bot.life_v1(packet=packet(request))
                        self.assertEqual(result['status'], 'submitted')
                        runner = self.bot._runners[result['task_id']]
                        if revoke == 'policy':
                            self.bot.config.life.enabled = False
                        with patch.object(mod, 'life_target_allowed', new=AsyncMock(return_value=False)):
                            self.bot._generation_slots.release()
                            await runner
                        task = self.bot._tasks[result['task_id']]
                        self.assertEqual(task.status, 'failed')
                        self.assertIn('未提交供应商', task.error)
                        self.bot._generate.assert_not_awaited()
                        self.bot._send.assert_not_awaited()
                    finally:
                        self.bot._generation_slots.release()

    async def test_life_cancelled_real_sqlite_worker_keeps_lock_and_all_budgets(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        calls = 0
        original = mod.consume_life
        def consume(*args, **kwargs):
            nonlocal calls
            calls += 1
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise AssertionError('thread barrier timed out')
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            body = self.life(directory)
            self.bot._generate = AsyncMock()
            self.bot._send = AsyncMock()
            with patch.object(mod, 'consume_life', side_effect=consume):
                pending = asyncio.create_task(self.bot.life_v1(packet=packet(body)))
                waiters = []
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    for _ in range(9):
                        pending.cancel()
                        await asyncio.sleep(0)
                    self.assertFalse(pending.done())
                    self.assertTrue(self.bot._life_lock.locked())
                    self.assertIn('chat', self.bot._reservations)
                    self.assertEqual(self.bot._life_pending, 1)
                    waiters = [asyncio.create_task(self.bot.life_v1(packet=packet(body)))
                               for _ in range(mod.MAX_ADMITTED - 1)]
                    await asyncio.sleep(0)
                    self.assertEqual((await self.bot.life_v1(packet=packet(body)))['status'], 'busy')
                    self.assertFalse((await self.bot.draw(prompt='scene', _host_context=tool_context()))['success'])
                    self.assertEqual(calls, 1)
                    for waiter in waiters:
                        waiter.cancel()
                    await asyncio.gather(*waiters, return_exceptions=True)
                    self.assertEqual(self.bot._life_pending, 1)
                    # Lifecycle shutdown must not deadlock on the held lock.
                    await asyncio.wait_for(self.bot.on_config_update('', {}, 'test'), 1)
                    self.assertFalse(pending.done())
                finally:
                    release.set()
                    await asyncio.gather(pending, *waiters, return_exceptions=True)
                self.assertTrue(pending.cancelled())
                self.assertFalse(self.bot._life_lock.locked())
                self.assertFalse(self.bot._reservations)
                self.assertEqual(self.bot._life_pending, 0)
            self.bot._generate.assert_not_awaited()
            self.bot._send.assert_not_awaited()
            with closing(sqlite3.connect(Path(directory, 'life-admission.db'))) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0], 1)
            self.assertEqual((await self.bot.life_v1(packet=packet(body)))['status'], 'unknown')

    async def test_worker_exception_never_overrides_external_cancellation(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        def fail():
            loop.call_soon_threadsafe(entered.set)
            release.wait(5)
            raise ValueError('synthetic worker failure')
        pending = asyncio.create_task(mod.thread_work(fail))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            pending.cancel()
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await pending
