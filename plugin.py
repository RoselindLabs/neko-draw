# -*- coding: utf-8 -*-
"""只用 OpenAI 兼容接口和 Gemini 的 MaiBot 生图插件。"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
import io
import math
from functools import wraps
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlsplit

import aiohttp
from PIL import Image

from maibot_sdk import API, Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType
from .source_images import ImageCache, extract, mime_type, resolve_edit_images
from .public_http import PublicClient, download_public, local_addresses
from .result_images import MAX_IMAGE_BYTES, decode_image, read_response, validate_image
from .life_authorization import authenticate as authenticate_life, consume as consume_life
from .life_authorization import target_allowed as life_target_allowed, target_policy as life_target_policy
from .self_reference import read_reference, portrait_prompt, portrait_edit_prompt
from .focus_scope import items_for_scope, planner_scopes
from .resource_budget import MAX_ADMITTED, MAX_GENERATING, MAX_PROMPT_CHARS, deadline_slot, image_work, thread_work


MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*]\(([^)\s]+)\)")
DATA_URL = re.compile(r"data:image/[\w.+-]+;base64,([A-Za-z0-9+/=\s]+)", re.IGNORECASE)


class GenerationUncertain(RuntimeError):
    """A request may have reached the supplier; do not retry or switch routes."""


class GenerationNotSubmitted(RuntimeError):
    """Explicit rejection or connection establishment failure; fallback is safe."""


class AdmissionExpired(RuntimeError):
    """Local work expired before its first generation attempt, never submitted."""


ASPECT_RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16")
ASPECT_RATIO_VALUES = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
}
OPENAI_STANDARD_SIZES = {
    "1:1": "1024x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
}
OPENAI_CUSTOM_SIZES = {
    "1:1": "1024x1024",
    "4:3": "1024x768",
    "3:4": "768x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "16:9": "1536x864",
    "9:16": "864x1536",
}
OPENAI_RATIO_MODELS = {"gpt-image-2", "gpt-image-2-2026-04-21", "gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5"}
GEMINI_RATIO_MODELS = {
    "gemini-2.5-flash-image", "gemini-2.5-flash-image-preview",
    "gemini-3-pro-image", "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview",
}


def ratio_capability(config, provider: str) -> str:
    """Configured contract, never inferred from auto's successful-route cache."""
    cfg = getattr(config, provider)
    if provider == "openai":
        if cfg.mode == "chat":
            return "chat"
        # auto includes Images even when chat was previously successful. Explicit
        # ratios require its known contract; this check never makes a probe.
        return "images" if cfg.model.strip() in OPENAI_RATIO_MODELS else "unknown"
    if cfg.mode == "openai":
        return "chat"
    return "native" if cfg.model.strip() in GEMINI_RATIO_MODELS else "unknown"


def task_route_ratio(task, capability: str) -> Optional[str]:
    ratio = getattr(task, 'aspect_ratio', None)
    if ratio and capability == "unknown":
        if not getattr(task, 'aspect_ratio_automatic', False):
            raise ValueError("当前路由模型未登记画幅能力，显式aspect_ratio/source未提交；不能猜测别名支持")
        task.aspect_ratio_feedback = "当前路由画幅能力未知，源比例未应用；沿用原配置及路由，不保证输出尺寸"
        return None
    return ratio


def normalize_aspect_ratio(value: Any, *, allow_source: bool = True) -> Optional[str]:
    """Validate the public value before any source lookup or supplier request."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("aspect_ratio 必须是 source 或有限比例枚举")
    text = value.strip().lower()
    if text == "source" and allow_source:
        return text
    if text in ASPECT_RATIOS:
        return text
    raise ValueError("aspect_ratio 仅支持1:1、4:3、3:4、3:2、2:3、16:9、9:16；source仅用于改图")


def nearest_ratio(ratio: float, choices=ASPECT_RATIOS) -> str:
    # Log distance treats portrait and landscape reciprocally.
    return min(choices, key=lambda item: abs(math.log(ASPECT_RATIO_VALUES[item] / ratio)))


def source_canvas(data: bytes) -> tuple[int, int]:
    """Read the already-bounded source canvas header; never decode animation frames."""
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def openai_ratio_size(model: str, ratio: str, canvas: Optional[tuple[int, int]] = None) -> tuple[str, str]:
    """Only known model contracts; do not infer support from an arbitrary alias."""
    if model in {"gpt-image-2", "gpt-image-2-2026-04-21"}:
        return OPENAI_CUSTOM_SIZES[ratio], ratio
    if model in {"gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5"}:
        target = canvas[0] / canvas[1] if canvas else ASPECT_RATIO_VALUES[ratio]
        nearest = nearest_ratio(target, OPENAI_STANDARD_SIZES)
        return OPENAI_STANDARD_SIZES[nearest], nearest
    raise ValueError("当前 Images 模型未登记尺寸支持，未提交；请管理员核验模型，不能保证所选画幅")


def chat_ratio_prompt(prompt: str, ratio: Optional[str]) -> str:
    return prompt + f"\n画面构图建议：宽高比 {ratio}（此聊天接口不保证输出尺寸）。" if ratio else prompt


DRAW_RATIO_HINT = "仅当前路由已登记画幅能力或明确chat模式时，用户指定优先，否则按画面主动选择aspect_ratio；能力未知应省略，保留配置且不保证画幅。chat仅构图建议，不保证尺寸。"
EDIT_RATIO_HINT = "用户指定画幅优先；省略时仅已知能力路由应用主源图比例，未知则保留配置并报告未应用。source/明确比例在未知能力路由会拒绝；角色参考不影响画幅，chat仅构图建议。"


def planner_ratio_hints(config) -> tuple[str, str]:
    # Describe only the first locally usable provider; a backup is checked when
    # reached, never allowed to veto a successful preferred provider.
    provider = config.general.provider
    cfg = getattr(config, provider)
    if not cfg.enabled or not cfg.api_key.strip():
        fallback = config.general.fallback_provider
        if fallback in {"openai", "gemini"}:
            provider = fallback
    if ratio_capability(config, provider) == "unknown":
        unavailable = ("当前有效路由画幅能力未登记，不要主动添加aspect_ratio。省略保留原配置/路由；"
                       "用户指定画幅时先说明无法保证，不把省略说成支持。显式比例/source会在请求前拒绝。")
        return unavailable, unavailable + "改图省略时源比例未应用；角色参考不决定画幅。"
    return ("用户指定画幅优先填aspect_ratio；未指定时按画面主动选择比例。省略沿用配置；chat仅构图建议，不保证尺寸。",
            "用户指定画幅优先填aspect_ratio；否则省略或source按第一张聊天源图画布匹配最接近比例，角色参考不影响；chat仅构图建议，不保证尺寸。")


def error_excerpt(text: str, limit: int) -> str:
    """Bound already-redacted diagnostics without silently hiding the cut."""
    marker = "…[已截断]"
    return text if len(text) <= limit else text[:limit - len(marker)] + marker


def admitted(method):
    """Reserve before any asynchronous source/portrait preprocessing."""
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        stream = self._stream_id(kwargs)
        generation = self._generation_id()
        if not self._current(generation):
            return {"success": False, "content": "插件已关闭，本次未提交。"}
        if not self._reserve(stream):
            active = self._active_task(stream)
            if active is not None:
                return {"success": False, "content": "本聊天已有任务，本次未重复提交；请等待或用neko_draw_status查看进度，不要重复提交。\n" + active.summary()}
            if stream in self._reservations:
                return {"success": False, "content": "本聊天已有请求正在准备，本次未重复提交；请等待原请求回执，创建任务后可用neko_draw_status查看进度，不要重复提交。"}
            return {"success": False, "content": "全局任务容量已满，本次未提交；可稍后按需重新请求。"}
        try:
            return await method(self, *args, **kwargs)
        except AdmissionExpired:
            return {"success": False, "content": "预处理或排队已过期，未提交供应商，请按需重新请求。"}
        finally:
            self._reservations.pop(stream, None)
    return wrapped


class PromptRejected(RuntimeError):
    """接口返回 HTTP 400，不自动重试；具体原因以实际错误说明为准。"""


class PluginSection(PluginConfigBase):
    __ui_label__ = "插件"

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default="1.1.0", description="配置版本")


class GeneralConfig(PluginConfigBase):
    __ui_label__ = "通用"

    provider: Literal["openai", "gemini"] = Field(default="openai", description="优先使用的接口")
    fallback_provider: Literal["none", "openai", "gemini"] = Field(default="gemini", description="明确未执行的失败后改试的接口；结果未知不改试。none 表示不备选")
    timeout: int = Field(default=180, ge=10, le=300, description="单次请求超时秒数")
    retry_times: int = Field(default=2, ge=0, le=5, description="同一个接口额外重试次数。0 表示不重试")
    retry_delay_seconds: float = Field(default=2.0, ge=0.5, le=30.0, description="首次重试前等待秒数，之后翻倍，最多 30 秒")


class CommandConfig(PluginConfigBase):
    __ui_label__ = "测试命令"

    allowed_user_ids: List[str] = Field(
        default_factory=list,
        description="仅允许使用 /draw 的 QQ 号。留空时 QQ 命令不生效",
    )
    allowed_telegram_user_ids: List[str] = Field(
        default_factory=list,
        description="允许使用 /draw 的 Telegram 数字用户 ID；独立于 QQ，留空时 Telegram 命令不生效",
    )


class OpenAIConfig(PluginConfigBase):
    __ui_label__ = "OpenAI"

    enabled: bool = Field(default=True, description="是否启用")
    base_url: str = Field(default="https://api.openai.com/v1", description="兼容接口地址")
    api_key: str = Field(default="", description="API Key")
    model: str = Field(default="gpt-image-2", description="生图模型")
    size: str = Field(default="1024x1024", description="draw未指定aspect_ratio时的Images默认尺寸，例如1024x1024；chat不保证尺寸")
    mode: Literal["auto", "images", "chat"] = Field(
        default="auto",
        description="auto=官方生图接口明确未执行时才改试聊天补全；成功响应后取图失败不改试",
    )
    proxy: str = Field(default="", description="HTTP 代理。国内中转留空")


class GeminiConfig(PluginConfigBase):
    __ui_label__ = "Gemini"

    enabled: bool = Field(default=False, description="是否启用")
    base_url: str = Field(default="https://generativelanguage.googleapis.com", description="Gemini 或兼容网关地址")
    api_key: str = Field(default="", description="API Key")
    model: str = Field(default="gemini-3.1-flash-image-preview", description="图片模型")
    aspect_ratio: str = Field(default="1:1", description="draw未指定aspect_ratio时的native默认宽高比；edit仅已知能力时匹配主源图，未知沿用配置；兼容chat不保证尺寸")
    mode: Literal["native", "openai"] = Field(
        default="native",
        description="native=generateContent；openai=把地址当 OpenAI 兼容网关",
    )
    proxy: str = Field(default="http://127.0.0.1:7890", description="访问 Google 用的代理")


class LifeConfig(PluginConfigBase):
    __ui_label__ = "Life配图授权"

    enabled: bool = Field(default=False, description="允许已配对Life服务自主配图，独立于普通生图工具",
                          json_schema_extra={"label": "允许Life自主配图", "hint": "仍须Life侧允许同一聊天；不影响普通生图工具，也不选择另一套生成模型。"})
    group_ids: List[str] = Field(default_factory=list, max_length=16, description="允许Life配图的QQ群号，最多16项",
                                json_schema_extra={"label": "允许的QQ群号", "hint": "每项纯数字群号；只匹配宿主已有QQ会话，不创建聊天。与高级会话ID取并集，撤权须一并移除旧ID。"})
    user_ids: List[str] = Field(default_factory=list, max_length=16, description="允许Life配图的私聊QQ号，最多16项",
                               json_schema_extra={"label": "允许私聊的QQ号", "hint": "只授权私聊，不授权该用户所在群。与群号分开填写，查询失败拒绝。"})
    account_id: str = Field(default="", max_length=20, description="可选机器人QQ账号限制",
                            json_schema_extra={"label": "高级：限定机器人QQ账号", "hint": "多个机器人时限定原账号，留空不额外限定；不是聊天对象QQ号。"})
    sessions: List[str] = Field(default_factory=list, description="高级兼容：精确宿主会话ID，与群号/QQ号取并集",
                               json_schema_extra={"label": "高级：允许的宿主会话ID", "hint": "日常填群号/QQ号即可；迁移后移除对应旧ID，否则只删号码不能撤权。所有名单都空拒绝。"})


class SelfPortraitConfig(PluginConfigBase):
    __ui_label__ = "自画像参考图"

    enabled: bool = Field(default=False, description="画自己时允许使用管理员配置的本地角色参考图",
                          json_schema_extra={"label": "使用自画像参考图", "hint": "开启后画自己优先参考图，不再先查记忆重建外貌；普通画图和Life无人场景不受影响。"})
    reference_path: str = Field(default="", max_length=4096, description="管理员指定的一张本地PNG/JPEG/WebP静态参考图的绝对路径",
                                json_schema_extra={"label": "本地参考图文件", "hint": "填写图片文件而非目录；最大10MiB/4000万像素。图片作为生图输入交给当前供应商，不在QQ单独发送；不要填写私密文件。"})


class PluginConfig(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    command: CommandConfig = Field(default_factory=CommandConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    self_portrait: SelfPortraitConfig = Field(default_factory=SelfPortraitConfig)
    life: LifeConfig = Field(default_factory=LifeConfig)


class DrawTask:
    """一次生图的进度。只放在内存里，重启后清空。"""

    def __init__(self, task_id: str, stream_id: str, prompt: str, user_id: str) -> None:
        now = time.time()
        self.task_id = task_id
        self.stream_id = stream_id
        self.prompt = prompt
        self.display_prompt = prompt
        self.user_id = user_id
        self.status = "queued"
        self.images: List[bytes] = []
        self.attempt = 0
        self.max_attempts = 1
        self.provider = ""
        self.error = ""
        self.started = now
        self.updated = now
        self.terminal_at: Optional[float] = None
        self.terminal_hint_used = False
        self.admission_deadline: Optional[float] = None
        self.life_policy = None
        self.aspect_ratio: Optional[str] = None
        self.aspect_ratio_automatic = False
        self.aspect_ratio_origin = "配置默认"
        self.aspect_ratio_feedback = ""
        self.source_canvas: Optional[tuple[int, int]] = None

    def touch(self, status: str, **fields: Any) -> None:
        if status in {'succeeded', 'failed', 'unknown'} and self.status not in {'succeeded', 'failed', 'unknown'}:
            self.terminal_at = time.monotonic()
            self.terminal_hint_used = False
        self.status = status
        self.updated = time.time()
        for key, value in fields.items():
            setattr(self, key, value)

    def summary(self) -> str:
        labels = {
            "queued": "排队",
            "running": "生成中",
            "retrying": "重试中",
            "succeeded": "已发送",
            "failed": "失败",
            "unknown": "结果未知（不自动重试或补发）",
        }
        lines = [
            f"任务 {self.task_id}",
            f"状态：{labels.get(self.status, self.status)}（第 {max(self.attempt, 1)}/{self.max_attempts} 次）",
            f"提示词：{self.display_prompt[:80]}",
        ]
        if self.provider:
            lines.append(f"接口：{self.provider}")
        if self.aspect_ratio:
            lines.append(f"画幅请求：{self.aspect_ratio}（{self.aspect_ratio_origin}）；" +
                         (self.aspect_ratio_feedback or "按后端能力映射，chat仅构图建议，不保证尺寸"))
        if self.error:
            lines.append(f"原因：{error_excerpt(self.error, 180)}")
        return "\n".join(lines)


class NekoDraw(MaiBotPlugin):
    """文生图与图生图，共用后台任务与配置的备选提供商。"""

    config_model = PluginConfig

    def __init__(self) -> None:
        super().__init__()
        self._session: Optional[aiohttp.ClientSession] = None
        self._tasks: Dict[str, DrawTask] = {}
        self._latest_by_stream: Dict[str, str] = {}
        self._runners: Dict[str, asyncio.Task] = {}
        self._preferred_openai_mode = ""
        self._preferred_openai_edit_mode = ""
        self._image_cache = ImageCache()
        self._life_lock = asyncio.Lock()
        self._life_closed = False
        self._generation = 0
        self._reservations = {}
        self._generation_slots = asyncio.Semaphore(MAX_GENERATING)

    def _generation_id(self):
        return getattr(self, '_generation', 0)

    def _current(self, generation):
        return (generation == self._generation_id() and not getattr(self, '_life_closed', False)
                and self.config.plugin.enabled)

    def _reserve(self, stream, *, deadline=None):
        if not hasattr(self, '_reservations'):
            self._reservations = {}
        occupied = set(self._reservations) | {t.stream_id for t in self._tasks.values()
                                               if t.status in {'queued', 'running', 'retrying'}}
        if stream in occupied or len(occupied) >= MAX_ADMITTED:
            return False
        self._reservations[stream] = (
            self._generation_id(),
            time.monotonic() + self.config.general.timeout if deadline is None else deadline,
        )
        return True

    @HookHandler("chat.receive.before_process", name="neko_source_images",
                  description="缓存本聊天入站图片/表情包源图供图生图使用", mode=HookMode.OBSERVE)
    async def cache_source_images(self, message: Any = None, stream_id: str = "", **kwargs: Any) -> None:
        generation = self._generation_id()
        if not self.config.plugin.enabled or not isinstance(message, dict):
            return
        stream = str(stream_id or kwargs.get("session_id") or message.get("session_id") or message.get("chat_id") or message.get("stream_id") or "")
        if stream:
            if not self._current(generation) or getattr(self, '_cache_pending', 0) >= MAX_ADMITTED:
                return
            self._cache_pending = getattr(self, '_cache_pending', 0) + 1
            try:
                images = await image_work(extract, message)
                if self._current(generation):
                    self._image_cache.put(stream, message, images=images)
            except (ValueError, TypeError):
                self.ctx.logger.debug("入站图片无法缓存，图生图时可尝试按消息 ID 读取")
            finally:
                self._cache_pending -= 1

    async def on_load(self) -> None:
        self._life_closed = False
        if not self.config.plugin.enabled:
            self.ctx.logger.info("生图插件已禁用")
            return
        self.ctx.logger.info(
            "生图插件已加载，优先=%s，备选=%s",
            self.config.general.provider,
            self.config.general.fallback_provider,
        )

    async def on_unload(self) -> None:
        self._life_closed = True
        self._generation = self._generation_id() + 1
        runners = list(self._runners.values())
        for runner in runners:
            runner.cancel()
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)
        # A task cancelled before its coroutine starts never executes finally.
        for task in self._tasks.values():
            if task.status in {"queued", "running", "retrying"}:
                task.touch("failed", error="插件已卸载，任务取消")
            task.images = []
        self._runners.clear()
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        del scope, config_data
        await self.on_unload()
        self._preferred_openai_mode = ""
        self._preferred_openai_edit_mode = ""
        self.ctx.logger.info("生图配置已热重载：version=%s", version)
        self._life_closed = False

    async def _http(self) -> aiohttp.ClientSession:
        if not self._current(self._generation_id()):
            raise asyncio.CancelledError
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.general.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    @staticmethod
    def _root(url: str) -> str:
        normalized = url.strip().rstrip("/")
        if normalized.endswith("/v1"):
            return normalized[:-3].rstrip("/")
        return normalized

    async def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str], proxy: str) -> Dict[str, Any]:
        session = await self._http()
        request_proxy = proxy.strip() or None
        async with session.post(url, json=payload, headers=headers, proxy=request_proxy) as response:
            return await self._generation_response(response)

    async def _generation_response(self, response):
        # Any successful submission is a one-way boundary, including body reads.
        try:
            raw = await read_response(response)
        except Exception as exc:
            raise GenerationUncertain("提交后响应读取失败；执行结果未知，不重新生成") from exc
        if response.status == 400:
            raise PromptRejected("HTTP 400：" + self._error_text(raw))
        if response.status in {401, 403, 404, 405, 415, 422, 429}:
            raise GenerationNotSubmitted(f"HTTP {response.status}：" + self._error_text(raw))
        if response.status != 200:
            raise GenerationUncertain(f"HTTP {response.status}：供应商是否已执行未知，不自动重试或切换接口")
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError('not an object')
            return data
        except Exception as exc:
            raise GenerationUncertain("HTTP 200 后响应 JSON 无效；不重新生成") from exc

    async def _openai_result(self, body, proxy):
        try:
            return (await self._images_from_openai_body(body, proxy))[0]
        except Exception as exc:
            raise GenerationUncertain("供应商已返回成功，图片读取/校验失败或无图片；不重新生成") from exc

    async def _download(self, url: str, proxy: str) -> bytes:
        async with PublicClient(timeout=self.config.general.timeout, proxy=proxy.strip() or None) as client:
            body, _, _ = await download_public(client, url, max_image_bytes=MAX_IMAGE_BYTES,
                                                blocked_ips=local_addresses())
        return await image_work(validate_image, body)

    async def _bytes_from_reference(self, value: str, proxy: str) -> Optional[bytes]:
        text = value.strip()
        if not text:
            return None
        if len(text) > 24 * 1024 * 1024:
            raise ValueError("生成图片引用超过大小限制")
        data_match = DATA_URL.search(text)
        if data_match:
            return await image_work(decode_image, data_match.group(1))
        if text.startswith("http://") or text.startswith("https://"):
            return await self._download(text, proxy)
        compact = re.sub(r"\s+", "", text)
        if (len(compact) > 200 or compact.startswith(('R0lGODdh', 'R0lGODlh'))) and re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
            try:
                decoded = await image_work(decode_image, compact)
            except Exception:
                return None
            if decoded.startswith((b"\x89PNG", b"\xff\xd8", b"RIFF", b"GIF87a", b"GIF89a")):
                return decoded
        return None

    async def _images_from_openai_body(self, body: Dict[str, Any], proxy: str) -> List[bytes]:
        images: List[bytes] = []
        data = body.get("data")
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                raw = item.get("b64_json") or item.get("url")
                if isinstance(raw, str):
                    parsed = await self._bytes_from_reference(raw, proxy)
                    if parsed:
                        return [parsed]
        if images:
            return images

        choices = body.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                blobs = [content] if isinstance(content, str) else []
                if isinstance(content, list):
                    blobs.extend(content)
                extra = message.get("images")
                if isinstance(extra, list):
                    blobs.extend(extra)
                for blob in blobs:
                    images.extend(await self._images_from_content(blob, proxy))
                    if images:
                        return images[:1]
        if not images:
            raise RuntimeError("响应里没有图片")
        return images

    async def _images_from_content(self, item: Any, proxy: str) -> List[bytes]:
        if isinstance(item, str):
            found: List[bytes] = []
            references = MARKDOWN_IMAGE.findall(item) or [item]
            for reference in references:
                parsed = await self._bytes_from_reference(reference, proxy)
                if parsed:
                    return [parsed]
            return found
        if not isinstance(item, dict):
            return []
        for key in ("b64_json", "url", "image_url", "data"):
            value = item.get(key)
            if isinstance(value, dict):
                value = value.get("url") or value.get("data")
            if isinstance(value, str):
                parsed = await self._bytes_from_reference(value, proxy)
                if parsed:
                    return [parsed]
        return []

    async def _openai_images(self, prompt: str, cfg: OpenAIConfig, images: Optional[List[bytes]] = None) -> bytes:
        if not cfg.api_key.strip():
            raise GenerationNotSubmitted("未配置 OpenAI API Key")
        root = self._root(cfg.base_url)
        if images:
            form = aiohttp.FormData()
            form.add_field("model", cfg.model.strip())
            form.add_field("prompt", prompt)
            form.add_field("n", "1")
            if cfg.size.strip():
                form.add_field("size", cfg.size.strip())
            for i, image in enumerate(images):
                mime = mime_type(image)
                form.add_field("image[]", image, filename=f"source-{i}.{mime.split('/')[-1]}", content_type=mime)
            session = await self._http()
            async with session.post(f"{root}/v1/images/edits", data=form,
                                    headers={"Authorization": f"Bearer {cfg.api_key.strip()}"},
                                    proxy=cfg.proxy.strip() or None) as response:
                body = await self._generation_response(response)
            return await self._openai_result(body, cfg.proxy)
        headers = {"Authorization": f"Bearer {cfg.api_key.strip()}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {"model": cfg.model.strip(), "prompt": prompt, "n": 1}
        if cfg.size.strip():
            payload["size"] = cfg.size.strip()
        if "gpt-image" not in cfg.model.lower():
            payload["response_format"] = "b64_json"
        body = await self._post_json(f"{root}/v1/images/generations", payload, headers, cfg.proxy)
        return await self._openai_result(body, cfg.proxy)

    async def _openai_chat(self, prompt: str, cfg: OpenAIConfig, images: Optional[List[bytes]] = None) -> bytes:
        if not cfg.api_key.strip():
            raise GenerationNotSubmitted("未配置 OpenAI API Key")
        root = self._root(cfg.base_url)
        headers = {"Authorization": f"Bearer {cfg.api_key.strip()}", "Content-Type": "application/json"}
        payload = {
            "model": cfg.model.strip(),
            "messages": [{"role": "user", "content": (
                [{"type": "text", "text": prompt}] + [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type(image)};base64,{base64.b64encode(image).decode('ascii')}"}}
                    for image in images
                ] if images else prompt
            )}],
            "stream": False,
        }
        body = await self._post_json(f"{root}/v1/chat/completions", payload, headers, cfg.proxy)
        return await self._openai_result(body, cfg.proxy)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        # Never infer permission to submit again from arbitrary exception text.
        # Connector establishment is handled separately; every post-submit read
        # failure (even a result URL returning 429) is GenerationUncertain.
        return isinstance(exc, GenerationNotSubmitted) and str(exc).startswith('HTTP 429：')

    async def _call_with_retry(self, task: Optional[DrawTask], label: str, func: Any) -> Any:
        attempts = self.config.general.retry_times + 1
        delay = self.config.general.retry_delay_seconds
        if task is not None:
            task.max_attempts = max(task.max_attempts, attempts)
        last: Exception = RuntimeError(f"{label} 没有执行")
        for attempt in range(1, attempts + 1):
            if task is not None:
                task.touch("running" if attempt == 1 else "retrying", attempt=attempt, provider=label)
            try:
                return await func()
            except GenerationUncertain:
                raise
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                if not isinstance(exc, aiohttp.ClientConnectorError):
                    raise GenerationUncertain("网络超时或连接中断，供应商执行结果未知；不自动重试或切换接口") from exc
                last = exc
                if attempt == attempts:
                    raise GenerationNotSubmitted("连接建立失败，未提交供应商") from exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
            except Exception as exc:
                last = exc
                if not self._retryable(exc) or attempt == attempts:
                    raise
                self.ctx.logger.info("%s 第 %s 次失败，%.1f 秒后重试：%s", label, attempt, delay, self._redact(str(exc))[:300])
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise last

    async def _openai(self, prompt: str, task: Optional[DrawTask] = None) -> bytes:
        cfg = self.config.openai
        if not cfg.enabled:
            raise GenerationNotSubmitted("OpenAI 未启用")
        if getattr(task, 'aspect_ratio', None) and not cfg.api_key.strip():
            raise GenerationNotSubmitted("未配置 OpenAI API Key")
        ratio = task_route_ratio(task, ratio_capability(self.config, "openai"))
        editing = bool(task and task.images)
        preferred = self._preferred_openai_edit_mode if editing else self._preferred_openai_mode
        if cfg.mode == "images":
            route_names = ["images"]
        elif cfg.mode == "chat":
            route_names = ["chat"]
        elif preferred == "chat":
            route_names = ["chat", "images"]
        else:
            route_names = ["images", "chat"]
        methods = [(name, self._openai_images if name == "images" else self._openai_chat) for name in route_names]
        errors: List[str] = []
        for name, method in methods:
            try:
                route_cfg = cfg
                route_prompt = prompt
                if ratio:
                    if name == "images":
                        size, mapped = openai_ratio_size(cfg.model.strip(), ratio, getattr(task, 'source_canvas', None))
                        route_cfg = cfg.model_copy(update={"size": size})
                        task.aspect_ratio_feedback = f"Images请求size={size}" + (f"，近似为{mapped}，非精确{ratio}" if mapped != ratio else "")
                    else:
                        route_prompt = chat_ratio_prompt(prompt, ratio)
                        task.aspect_ratio_feedback = "chat仅提示词构图建议，不保证输出尺寸"
                image = await self._call_with_retry(task, f"openai/{name}",
                                                    lambda method=method, route_prompt=route_prompt, route_cfg=route_cfg:
                                                    method(route_prompt, route_cfg, task.images if task else None))
                if editing:
                    self._preferred_openai_edit_mode = name
                else:
                    self._preferred_openai_mode = name
                return image
            except (PromptRejected, GenerationUncertain):
                raise
            except GenerationNotSubmitted as exc:
                errors.append(str(exc))
                if cfg.mode != "auto":
                    break
        raise GenerationNotSubmitted("；".join(errors) or "OpenAI 生图失败")

    async def _gemini_native(self, prompt: str, images: Optional[List[bytes]] = None, aspect_ratio: Optional[str] = None) -> bytes:
        cfg = self.config.gemini
        ratio = cfg.aspect_ratio.strip() if aspect_ratio is None else normalize_aspect_ratio(aspect_ratio, allow_source=False)
        if aspect_ratio is not None and cfg.model.strip() not in GEMINI_RATIO_MODELS:
            raise ValueError("当前 Gemini native 模型未登记画幅支持，未提交；请管理员核验模型")
        if not cfg.api_key.strip():
            raise GenerationNotSubmitted("未配置 Gemini API Key")
        root = cfg.base_url.strip().rstrip("/")
        if root.endswith("/v1beta"):
            url = f"{root}/models/{cfg.model.strip()}:generateContent"
        else:
            url = f"{root}/v1beta/models/{cfg.model.strip()}:generateContent"
        host = urlsplit(root).hostname or ""
        headers = {"Content-Type": "application/json"}
        if "googleapis.com" in host:
            headers["x-goog-api-key"] = cfg.api_key.strip()
        else:
            headers["Authorization"] = f"Bearer {cfg.api_key.strip()}"
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}] + [
                {"inlineData": {"mimeType": mime_type(image), "data": base64.b64encode(image).decode("ascii")}}
                for image in (images or [])
            ]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        if ratio:
            payload["generationConfig"]["imageConfig"] = {"aspectRatio": ratio}
        body = await self._post_json(url, payload, headers, cfg.proxy)
        try:
            return await self._gemini_result(body)
        except Exception as exc:
            raise GenerationUncertain("供应商已返回成功，Gemini 图片读取/校验失败或无图片；不重新生成") from exc

    async def _gemini_result(self, body):
        images: List[bytes] = []
        for candidate in body.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData") or part.get("inline_data") or {}
                raw = inline.get("data") if isinstance(inline, dict) else None
                if isinstance(raw, str) and raw:
                    return await image_work(decode_image, raw)
        if not images:
            raise RuntimeError("Gemini 响应里没有图片")
        return images[0]

    async def _gemini(self, prompt: str, task: Optional[DrawTask] = None) -> bytes:
        cfg = self.config.gemini
        if not cfg.enabled:
            raise GenerationNotSubmitted("Gemini 未启用")
        if getattr(task, 'aspect_ratio', None) and not cfg.api_key.strip():
            raise GenerationNotSubmitted("未配置 Gemini API Key")
        ratio = task_route_ratio(task, ratio_capability(self.config, "gemini"))
        if cfg.mode == "openai":
            mirrored = OpenAIConfig(
                enabled=True,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                model=cfg.model,
                size="",
                mode="chat",
                proxy=cfg.proxy,
            )
            if ratio:
                task.aspect_ratio_feedback = "Gemini兼容chat仅提示词构图建议，不保证输出尺寸"
            return await self._call_with_retry(task, "gemini/openai", lambda: self._openai_chat(chat_ratio_prompt(prompt, ratio), mirrored, task.images if task else None))
        if ratio:
            task.aspect_ratio_feedback = f"Gemini native请求aspectRatio={ratio}（不是输出像素验收）"
        extra = {"aspect_ratio": ratio} if ratio else {}
        return await self._call_with_retry(task, "gemini", lambda: self._gemini_native(prompt, task.images if task else None, **extra))

    async def _generate(self, prompt: str, task: Optional[DrawTask] = None) -> Tuple[bytes, str]:
        order = [self.config.general.provider]
        fallback = self.config.general.fallback_provider
        if fallback != "none" and fallback not in order:
            order.append(fallback)
        errors: List[str] = []
        for name in order:
            try:
                if name == "openai":
                    return await self._openai(prompt, task), name
                if name == "gemini":
                    return await self._gemini(prompt, task), name
            except (PromptRejected, GenerationUncertain):
                raise
            except GenerationNotSubmitted as exc:
                message = self._redact(str(exc))
                errors.append(f"{name}: {message}")
                self.ctx.logger.warning("生图接口失败：%s", errors[-1][:500])
        raise RuntimeError("；".join(errors) or "没有可用的生图接口")

    async def _send(self, image: bytes, stream_id: str, prompt: str) -> None:
        generation = self._generation_id()
        await image_work(validate_image, image)
        if not self._current(generation):
            raise asyncio.CancelledError
        encoded = base64.b64encode(image).decode("ascii")
        sent = await self.ctx.send.image(
            encoded,
            stream_id,
            processed_plain_text=prompt[:80],
            sync_to_maisaka_history=True,
        )
        if not sent:
            raise RuntimeError("图片没有发送出去")

    def _stream_id(self, kwargs: Dict[str, Any]) -> str:
        return str(kwargs.get("stream_id") or kwargs.get("session_id") or "").strip()

    def _remember_task(self, task: DrawTask) -> None:
        self._tasks[task.task_id] = task
        self._latest_by_stream[task.stream_id] = task.task_id
        if len(self._tasks) > 30:
            oldest = sorted(self._tasks.values(), key=lambda item: item.started)[:-30]
            for item in oldest:
                if item.status in {"queued", "running", "retrying"}:
                    continue
                self._tasks.pop(item.task_id, None)
                if self._latest_by_stream.get(item.stream_id) == item.task_id:
                    self._latest_by_stream.pop(item.stream_id, None)

    def _active_task(self, stream_id: str) -> Optional[DrawTask]:
        task_id = self._latest_by_stream.get(stream_id, "")
        task = self._tasks.get(task_id)
        if task is not None and task.status in {"queued", "running", "retrying"}:
            return task
        return None

    def _status_text(self, stream_id: str) -> str:
        task_id = self._latest_by_stream.get(stream_id, "")
        task = self._tasks.get(task_id)
        if task is None:
            return "当前没有本聊天可核实的内存任务记录；可能尚未创建、已重载或已淘汰。不能据历史提交回执判断仍在排队或已成功。"
        return task.summary()

    @API('life_v1', version='1', public=True, description='签名Life配图任务，只接收独立授权且配对的会话')
    async def life_v1(self, packet=None, **kwargs):
        del kwargs
        # Bound even the callers waiting for signed durable admission/Host lookup.
        if getattr(self, '_life_pending', 0) >= MAX_ADMITTED:
            return {'success': False, 'status': 'busy'}
        self._life_pending = getattr(self, '_life_pending', 0) + 1
        deadline = time.monotonic() + self.config.general.timeout
        try:
            return await self._admit_life(packet, deadline)
        except TimeoutError:
            return {'success': False, 'status': 'expired'}
        finally:
            self._life_pending -= 1

    async def _admit_life(self, packet, deadline):
        generation = self._generation_id()
        async with deadline_slot(self._life_lock, deadline):
            if not self._current(generation) or not self.config.life.enabled:
                return {'success': False, 'status': 'disabled'}
            if time.monotonic() >= deadline:
                return {'success': False, 'status': 'expired'}
            reserved = False
            session = ''
            try:
                body, nonce = authenticate_life(packet, self.ctx.paths.data_dir)
                session = body['session_id']
                policy = life_target_policy(self.config.life)
                if not await life_target_allowed(self.ctx, session, policy):
                    return {'success': False, 'status': 'session_denied'}
                if not self._current(generation) or not self._life_policy_current(policy):
                    return {'success': False, 'status': 'cancelled'}
                if time.monotonic() >= deadline:
                    return {'success': False, 'status': 'expired'}
                reserved = self._reserve(session, deadline=deadline)
                fresh = await thread_work(consume_life, self.ctx.paths.data_dir, body, nonce, allow_new=reserved)
                task_id = body['request_id']
                if not fresh:
                    task = self._tasks.get(task_id)
                    return {'success': True, 'status': task.status if task else 'unknown', 'task_id': task_id}
                if not self._current(generation) or not self._life_policy_current(policy):
                    return {'success': False, 'status': 'cancelled'}
                task = DrawTask(task_id, session, body['prompt'], 'neko.life')
                task.generation = generation
                task.admission_deadline = deadline
                task.life_policy = policy
                task.max_attempts = self.config.general.retry_times + 1
                self._remember_task(task)
                if time.monotonic() >= deadline:
                    task.touch('failed', error='准入等待已过期，未提交供应商')
                    return {'success': False, 'status': 'expired', 'task_id': task_id}
                self._runners[task_id] = asyncio.create_task(self._run_task(task))
                return {'success': True, 'status': 'submitted', 'task_id': task_id}
            except ValueError as exc:
                return {'success': False, 'status': 'busy' if str(exc) == 'admission_busy' else 'admission_rejected'}
            except Exception:
                return {'success': False, 'status': 'admission_rejected'}
            finally:
                if reserved:
                    self._reservations.pop(session, None)

    def _life_policy_current(self, policy):
        return (not self._life_closed and self.config.plugin.enabled and self.config.life.enabled
                and life_target_policy(self.config.life) == policy)

    def _start_task(self, stream_id: str, prompt: str, user_id: str, images: Optional[List[bytes]] = None,
                    *, aspect_ratio: Optional[str] = None, aspect_ratio_origin: str = "显式参数",
                    source_canvas: Optional[tuple[int, int]] = None, aspect_ratio_automatic: bool = False) -> DrawTask:
        aspect_ratio = normalize_aspect_ratio(aspect_ratio, allow_source=False)
        if not self._current(self._generation_id()):
            raise asyncio.CancelledError
        if len(prompt) > MAX_PROMPT_CHARS + 2000:
            raise ValueError('提示词过长')
        active = self._active_task(stream_id)
        if active is not None:
            return active
        owned = stream_id not in getattr(self, '_reservations', {})
        if owned and not self._reserve(stream_id):
            raise ValueError('全局任务容量已满，未提交')
        generation, deadline = self._reservations[stream_id]
        if generation != self._generation_id() or time.monotonic() >= deadline:
            if owned:
                self._reservations.pop(stream_id, None)
            raise AdmissionExpired
        task = DrawTask(uuid.uuid4().hex[:8], stream_id, prompt, user_id)
        task.generation = self._generation_id()
        task.admission_deadline = deadline
        task.images = images or []
        task.aspect_ratio = aspect_ratio
        task.aspect_ratio_automatic = aspect_ratio_automatic
        task.aspect_ratio_origin = aspect_ratio_origin if aspect_ratio else "配置默认"
        if aspect_ratio_automatic and ratio_capability(self.config, self.config.general.provider) == "unknown":
            task.aspect_ratio_feedback = "首选路由画幅能力未知，源比例未应用；沿用原配置及路由，不保证输出尺寸"
        task.source_canvas = source_canvas
        task.max_attempts = self.config.general.retry_times + 1
        self._remember_task(task)
        self._runners[task.task_id] = asyncio.create_task(self._run_task(task))
        if owned:
            self._reservations.pop(stream_id, None)
        return task

    async def _run_task(self, task: DrawTask) -> None:
        sending = False
        generation = getattr(task, 'generation', self._generation_id())
        try:
            if not hasattr(self, '_generation_slots'):
                self._generation_slots = asyncio.Semaphore(MAX_GENERATING)
            deadline = task.admission_deadline
            if deadline is None:  # Legacy/in-memory test tasks, not a new admission.
                deadline = time.monotonic() + self.config.general.timeout
                task.admission_deadline = deadline
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AdmissionExpired
            try:
                async with asyncio.timeout(remaining):
                    await self._generation_slots.acquire()
            except TimeoutError as exc:
                raise AdmissionExpired from exc
            try:
                if not self._current(generation):
                    raise asyncio.CancelledError
                if task.life_policy is not None:
                    if not self._life_policy_current(task.life_policy):
                        raise RuntimeError('Life授权已撤销，未提交供应商')
                    if not await life_target_allowed(self.ctx, task.stream_id, task.life_policy):
                        raise RuntimeError('Life目标授权已失效，未提交供应商')
                    if not self._current(generation):
                        raise asyncio.CancelledError
                    if not self._life_policy_current(task.life_policy):
                        raise RuntimeError('Life授权已撤销，未提交供应商')
                if time.monotonic() >= deadline:
                    raise AdmissionExpired
                image, provider = await self._generate(task.prompt, task)
                if not self._current(generation):
                    raise asyncio.CancelledError
                sending = True
                await self._send(image, task.stream_id, task.display_prompt)
            finally:
                self._generation_slots.release()
            task.touch("succeeded", provider=provider, error="")
            self.ctx.logger.info("任务 %s 已用 %s 发送", task.task_id, provider)
        except asyncio.CancelledError:
            uncertain = sending or task.status in {"running", "retrying"}
            task.touch("unknown" if uncertain else "failed", error="任务已取消；已发起的外部执行或发送结果未知，不自动重试或补发" if uncertain else "插件已卸载，任务取消")
            raise
        except AdmissionExpired:
            task.touch('failed', error='预处理或排队已过期，未提交供应商，请按需重新请求')
        except GenerationUncertain as exc:
            task.touch("unknown", error=self._redact(str(exc)))
        except PromptRejected as exc:
            message = self._policy_feedback(task.prompt, str(exc))
            task.touch("failed", error=message)
            self.ctx.logger.info("任务 %s 被接口以 400 拒绝：%s", task.task_id, message[:300])
        except Exception as exc:
            message = self._redact(str(exc))
            task.touch("unknown" if sending else "failed", error=("图片发送结果未知，不自动补发。" + message) if sending else message)
            self.ctx.logger.error("任务 %s 失败：%s", task.task_id, message[:500])
        finally:
            task.images = []
            self._runners.pop(task.task_id, None)

    @HookHandler(
        "maisaka.planner.before_request",
        name="neko_draw_planner",
        description="生图路由与自画像记忆提示",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        error_policy=ErrorPolicy.SKIP,
    )
    async def configure_planner(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        changed = self._inject_task_status(kwargs)
        if self.config.plugin.enabled:
            changed = self._rewrite_draw_tool_hints(kwargs) or changed
        items = kwargs.get("items")
        messages = kwargs.get("messages")
        focused = not self._stream_id(kwargs)
        draw_scopes = [sid for sid in planner_scopes('', kwargs.get('tool_definitions'))
                       if self._is_draw_turn(items_for_scope(items, sid), None)] if focused else []
        draw_turn = bool(draw_scopes) if focused else self._is_draw_turn(items if isinstance(items, list) else None, messages if isinstance(messages, list) else None)
        if self.config.plugin.enabled and draw_turn:
            instruction = self._planner_instruction(self.config.self_portrait.enabled, planner_ratio_hints(self.config))
            if focused:
                instruction = '仅针对以下目标聊天的本次绘图请求（不是自动执行授权）：' + json.dumps(draw_scopes) + '\n' + instruction
            draw_changed = False
            if isinstance(items, list) and instruction not in "\n".join(self._item_text(item) for item in items):
                items.append(self._system_item(instruction))
                kwargs["items"] = items
                draw_changed = True
            if isinstance(messages, list) and instruction not in "\n".join(self._message_text(message) for message in messages):
                messages.append({"role": "system", "content": instruction})
                kwargs["messages"] = messages
                draw_changed = True
            if draw_changed:
                changed = True
                self.ctx.logger.debug("已注入条件式生图说明（不表示当前有生图任务）")
        if not changed:
            return None
        return {"action": "continue", "modified_kwargs": kwargs}

    def _redact(self, message: str) -> str:
        for secret in sorted((self.config.openai.api_key.strip(), self.config.gemini.api_key.strip()), key=len, reverse=True):
            if secret:
                message = message.replace(secret, "***")
        return message

    def _api_error_detail(self, raw: str) -> str:
        body = (raw.split("：", 1)[-1] if raw.startswith('HTTP ') else raw).strip()
        try:
            data = json.loads(body)
        except Exception:
            return ""
        return self._decoded_error_detail(data)

    def _decoded_error_detail(self, data) -> str:
        if not isinstance(data, dict):
            return ""
        error = data.get("error")
        if isinstance(error, dict):
            return self._redact(str(error.get("message") or error.get("code") or error.get("type") or "").strip())
        if isinstance(error, str):
            return self._redact(error.strip())
        return self._redact(str(data.get("message") or "").strip())

    def _error_text(self, raw: bytes) -> str:
        text = raw.decode('utf-8', 'replace')  # read_response already bounds bytes.
        try:
            data = json.loads(text)
        except Exception:
            return error_excerpt(self._redact(text), 500)
        else:
            # Decode JSON escapes before redaction/truncation; expose only a
            # whitelisted detail, never serialize the supplier's entire object.
            detail = error_excerpt(self._decoded_error_detail(data) or '接口未提供可用错误说明', 450)
            text = json.dumps({'error': {'message': detail}}, ensure_ascii=False)
            # Keep the bounded envelope valid JSON so later parsing cannot lose
            # the useful detail. Escaped quotes/control characters count too.
            if len(text) > 500:
                low, high = len("…[已截断]"), len(detail)
                while low < high:
                    middle = (low + high + 1) // 2
                    candidate = json.dumps({'error': {'message': error_excerpt(detail, middle)}}, ensure_ascii=False)
                    if len(candidate) <= 500:
                        low = middle
                    else:
                        high = middle - 1
                text = json.dumps({'error': {'message': error_excerpt(detail, low)}}, ensure_ascii=False)
        return self._redact(text)

    def _policy_feedback(self, prompt: str, raw: str) -> str:
        detail = self._redact(self._api_error_detail(self._redact(raw)))
        generic = detail.lower() in {"", "openai_error", "upstream_error", "invalid_request_error", "bad_request", "unknown_error", "error"}
        lines = ["HTTP 400；接口说明：" + error_excerpt(detail or "未提供可用说明", 300)]
        if generic:
            lines.append("原因不明。")
        lines.append("本次失败；不凭400推断内容审核或源图原因，不自动重试。")
        return "\n".join(lines)

    def _inject_task_status(self, kwargs: Dict[str, Any]) -> bool:
        """In-flight truth; terminal correction at most once within 15 minutes.

        Always remove our old system hint, even if no new hint is warranted.
        A consumed hint means prepared for RPC, not delivered/read by the model.
        """
        stream_id = self._stream_id(kwargs)
        marker = "[neko.draw 当前任务状态]"
        if not stream_id:
            changed = False
            for key, role_key, role_value, read in (
                ('items', 'item_type', 'SystemMessageItem', self._item_text),
                ('messages', 'role', 'system', self._message_text),
            ):
                values = kwargs.get(key)
                if isinstance(values, list):
                    fresh = [value for value in values if not (isinstance(value, dict)
                             and value.get(role_key) == role_value and read(value).startswith(marker))]
                    changed |= len(fresh) != len(values)
                    kwargs[key] = fresh
            if not any(isinstance(kwargs.get(key), list) for key in ('items', 'messages')):
                return changed
            for scope in planner_scopes('', kwargs.get('tool_definitions')):
                scoped = {'session_id': scope, 'items': []}
                if not self._inject_task_status(scoped):
                    continue
                for item in scoped['items']:
                    notice = marker + '\n目标 session_id=' + json.dumps(scope) + '\n' + self._item_text(item)[len(marker):]
                    if isinstance(kwargs.get('items'), list):
                        kwargs['items'].append(self._system_item(notice))
                    if isinstance(kwargs.get('messages'), list):
                        kwargs['messages'].append({'role': 'system', 'content': notice})
                    changed = True
            return changed
        items = kwargs.get("items")
        messages = kwargs.get("messages")
        changed = False
        for key, role_key, role_value, read in (
            ('items', 'item_type', 'SystemMessageItem', self._item_text),
            ('messages', 'role', 'system', self._message_text),
        ):
            values = kwargs.get(key)
            if isinstance(values, list):
                fresh = [value for value in values if not (isinstance(value, dict)
                         and value.get(role_key) == role_value and read(value).startswith(marker))]
                if len(fresh) != len(values):
                    kwargs[key] = fresh
                    changed = True
        items, messages = kwargs.get('items'), kwargs.get('messages')
        if not stream_id or not self.config.plugin.enabled:
            return changed
        task_id = self._latest_by_stream.get(stream_id, "")
        task = self._tasks.get(task_id)
        if task is None:
            return changed
        else:
            terminal = task.status in {'succeeded', 'failed', 'unknown'}
            if terminal and (task.terminal_hint_used or task.terminal_at is None
                             or time.monotonic() >= task.terminal_at + 900):
                return changed
            if task.status not in {'queued', 'running', 'retrying', 'succeeded', 'failed', 'unknown'}:
                return changed
            states = {"queued": "已提交，尚未开始", "running": "正在执行", "retrying": "正在重试",
                       "succeeded": "已完成，图片发送接口已确认发送", "failed": "已失败或取消，不在排队，也不会继续执行",
                       "unknown": "外部执行或图片发送结果未知，不自动重试、换工具重画或补发；不要声称成功或确定失败"}
            state = f"任务 {task.task_id}：{states.get(task.status, '状态未知')}。"
            if task.status == "succeeded":
                state += " 不要再说还在画或等待出图；无需重复发送或仅为宣告完成而刷屏。"
            if task.status == 'failed':
                state += " 错误摘录可按需查询 neko_draw_status（长说明会标记截断，并非完整接口原文）；不要仅因这条状态重复道歉或重新生成。"
            if task.aspect_ratio:
                state += f" 画幅请求{task.aspect_ratio}；" + (task.aspect_ratio_feedback or "按后端映射，chat仅构图建议，不保证尺寸") + "。"
        notice = marker + "\n" + state + "\n这是本聊天在本次规划请求前核实的状态，优先于旧的排队/提交回执；不构成新的生图请求。"
        inserted = False
        if isinstance(items, list):
            fresh = [item for item in items if not (
                isinstance(item, dict) and item.get("item_type") == "SystemMessageItem" and self._item_text(item).startswith(marker))]
            fresh.append(self._system_item(notice))
            kwargs["items"] = fresh
            changed = True
            inserted = True
        if isinstance(messages, list):
            fresh = [message for message in messages if not (
                isinstance(message, dict) and message.get("role") == "system" and self._message_text(message).startswith(marker))]
            fresh.append({"role": "system", "content": notice})
            kwargs["messages"] = fresh
            changed = True
            inserted = True
        if inserted and terminal:
            task.terminal_hint_used = True
        return changed

    @staticmethod
    def _message_text(message: Any) -> str:
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "\n".join(parts)
        return ""

    @staticmethod
    def _item_text(item: Any) -> str:
        if not isinstance(item, dict):
            return ""
        parts = item.get("parts")
        if not isinstance(parts, list):
            return str(item.get("text") or item.get("content") or "")
        chunks: List[str] = []
        for part in parts:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                chunks.append(str(part.get("text") or ""))
        return "\n".join(chunks)

    def _is_draw_turn(self, items: Optional[List[Any]], messages: Optional[List[Any]]) -> bool:
        texts: List[str] = []
        if items:
            user_items = [
                item for item in items if isinstance(item, dict) and item.get("item_type") == "UserMessageItem"
            ]
            texts.extend(self._item_text(item) for item in user_items[-1:])
        if messages:
            user_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "user"]
            texts.extend(self._message_text(message) for message in user_messages[-1:])
        text = "\n".join(texts)
        if re.search(r"(?:不要|别|不用|停止|取消|不需要).{0,8}(?:画|生图|生成图片)", text):
            return False
        return bool(re.search(
            r"(?:^|\s)/(?:draw|生图|画图)\s+|(?:请|帮我|给我|能不能|可以).{0,8}(?:画|生成.{0,3}图)|"
            r"(?:画一[张幅个]|画你自己|画个自己|画一下|生成一[张幅].{0,4}图)|\bdraw\s+(?:me|a|an|yourself)\b",
            text, re.IGNORECASE,
        ))

    @staticmethod
    def _system_item(text: str) -> Dict[str, Any]:
        return {
            "item_type": "SystemMessageItem",
            "meta": {
                "item_id": uuid.uuid4().hex,
                "logical_turn_id": None,
                "timestamp": datetime.now().isoformat(),
            },
            "parts": [{"type": "text", "text": text}],
        }

    @staticmethod
    def _planner_instruction(self_reference_enabled: bool = False, ratio_hints=None) -> str:
        draw_hint, edit_hint = ratio_hints or (DRAW_RATIO_HINT, EDIT_RATIO_HINT)
        if self_reference_enabled:
            return (
                "仅当前用户确实要求画图时才使用生图工具；历史提及、闲聊和取消不算。"
                "画你自己、你的自拍、自画像或以你为主角用neko_draw并设use_self_reference=true，"
                "插件附管理员角色参考图，不必先调用query_memory，prompt只写本次场景、动作、服装或画风，不编造消息ID或文件路径。"
                "修改聊天源图或GIF动图用neko_edit_image；上下文预览可能压缩，不能称为原始文件。GIF编辑仅补充高分辨率高帧要求，不保证效果；需要机器人外貌时同样设use_self_reference=true，保留聊天源图并追加角色参考。"
                "用户自拍、其他人物、无人场景和Life插画不附角色参考。参考图不可用时如实说明，不退回纯文生图。"
                + draw_hint + edit_hint
            )
        return (
            "仅当前用户确实要求画图时才使用生图工具；历史提及、闲聊和取消不算。"
            "修改聊天源图或GIF动图用neko_edit_image；上下文预览可能压缩，不能称为原始文件。GIF编辑仅补充高分辨率高帧要求，不保证效果。"
            "自画像先调用 query_memory 查长期记忆中的外貌，再把查到的特征用于neko_draw；未知特征不当作已有记忆，用户要求的新造型作为本次设定。"
            "其他文生图按用户要求调用neko_draw；自然语言、标签或混合描述均可。"
            + draw_hint + edit_hint
        )

    @staticmethod
    def _draw_tool_hints(self_reference_enabled: bool) -> Dict[str, Dict[str, str]]:
        """按当前自画像配置给模型看的工具描述；静态注册文本无法随配置变化，由每轮 hook 改写。"""
        if self_reference_enabled:
            return {
                "neko_draw": {
                    "description": (
                        "画图/文生图生成新图；修改聊天原图用neko_edit_image。"
                        "画机器人自己（含自拍、自画像）设use_self_reference=true，无需先查query_memory。"
                    ),
                    "flag_description": "仅画机器人自己时设true，插件附管理员角色参考图；prompt只写场景、动作、服装或画风。参考图不可用则失败，不退回纯文生图。其他人物、用户自拍或无人场景保持false；不接受路径或消息ID。",
                },
                "neko_edit_image": {
                    "description": "改图/图生图：修改聊天图片或GIF，如换背景、换装、重绘。需要机器人角色外貌时设use_self_reference=true。",
                    "flag_description": "修改涉及机器人角色且需核对外貌时设true；最多3张聊天源图在前，1张角色参考附最后，不替换源图，也不把所有人物改成该角色。参考不可用则失败，不降级。",
                },
            }
        return {
            "neko_draw": {
                "description": (
                    "画图/文生图生成新图；修改聊天原图用neko_edit_image。"
                    "画机器人自己（含自拍、自画像）先用query_memory查长期记忆外貌，写入prompt，不编造未知特征。"
                ),
                "flag_description": "当前未配置角色参考图，保持false；画机器人自己先用query_memory查记忆外貌。",
            },
            "neko_edit_image": {
                "description": "改图/图生图：修改聊天图片或GIF，如换背景、换装、重绘。当前未配置角色参考图；涉及机器人外貌时先用query_memory查记忆并写入prompt。",
                "flag_description": "当前未配置角色参考图，保持false。",
            },
        }

    def _rewrite_draw_tool_hints(self, kwargs: Dict[str, Any]) -> bool:
        """把neko_draw/neko_edit_image的工具与参数提示改写为当前自画像配置的版本，不动其他工具。"""
        definitions = kwargs.get("tool_definitions")
        if not isinstance(definitions, list) or not definitions:
            return False
        hints = self._draw_tool_hints(bool(self.config.self_portrait.enabled))
        draw_hint, edit_hint = planner_ratio_hints(self.config)
        rewritten: List[Any] = []
        changed = False
        for tool in definitions:
            function = tool.get("function") if isinstance(tool, dict) else None
            hint = hints.get(function.get("name")) if isinstance(function, dict) else None
            if hint is None:
                rewritten.append(tool)
                continue
            new_function = dict(function)
            ratio_hint = draw_hint if function["name"] == "neko_draw" else edit_hint
            new_function["description"] = hint["description"] + "画幅选择见aspect_ratio参数说明。"
            parameters = function.get("parameters")
            properties = parameters.get("properties") if isinstance(parameters, dict) else None
            flag = properties.get("use_self_reference") if isinstance(properties, dict) else None
            if isinstance(flag, dict):
                new_parameters = dict(parameters)
                new_properties = dict(properties)
                new_flag = dict(flag)
                new_flag["description"] = hint["flag_description"]
                new_properties["use_self_reference"] = new_flag
                new_parameters["properties"] = new_properties
                new_function["parameters"] = new_parameters
            ratio_parameter = properties.get("aspect_ratio") if isinstance(properties, dict) else None
            if isinstance(ratio_parameter, dict):
                new_parameters = dict(new_function.get("parameters", parameters))
                new_properties = dict(new_parameters["properties"])
                new_properties["aspect_ratio"] = {**ratio_parameter, "description": ratio_hint}
                new_parameters["properties"] = new_properties
                new_function["parameters"] = new_parameters
            rewritten.append({"type": "function", "function": new_function})
            changed = True
        if changed:
            kwargs["tool_definitions"] = rewritten
        return changed

    @Tool(
        "neko_draw",
        trusted_context="neko.tools.context.v1",
        brief_description="画图/文生图生成新图，含自拍、自画像；已知画幅能力时用户指定优先，否则按画面选比例。",
        detailed_description=(
            "生成全新图片；修改聊天中已有图片用neko_edit_image。"
            "画机器人自己的提示按当前自画像配置处理：已配置参考图时设use_self_reference=true由插件附图，"
            "未配置时先用query_memory查记忆中的外貌。"
            "自然语言、标签或混合描述均可，无需列全画面要素。"
        ),
        parameters=[
            ToolParameterInfo(
                name="prompt",
                param_type=ToolParamType.STRING,
                description="主体、场景、动作、服装或画风；自然语言/标签均可",
                required=True,
            ),
            ToolParameterInfo(name="use_self_reference", param_type=ToolParamType.BOOLEAN,
                              description="是否画机器人自己，按当前自画像配置的说明使用；不接受路径或消息ID。", required=False),
            ToolParameterInfo(name="aspect_ratio", param_type=ToolParamType.STRING,
                              description=DRAW_RATIO_HINT, enum_values=list(ASPECT_RATIOS), required=False),
        ],
    )
    async def draw(self, prompt: str = "", use_self_reference: bool = False, aspect_ratio: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        identity = self._tool_identity(kwargs.get("_host_context"))
        if identity is None:
            return {"success": False, "content": "可信聊天身份缺失或矛盾，本次未提交；请管理员核对Host可信上下文与会话路由，不要自行填写身份重试。"}
        return await self._draw(prompt, use_self_reference, aspect_ratio, **identity)

    @admitted
    async def _draw(self, prompt: str = "", use_self_reference: bool = False, aspect_ratio: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        generation = self._generation_id()
        try:
            ratio = normalize_aspect_ratio(aspect_ratio, allow_source=False)
        except ValueError as exc:
            return {"success": False, "content": str(exc) + "，未提交。"}
        if not isinstance(prompt, str) or type(use_self_reference) is not bool:
            return {"success": False, "content": "提示词或自画像参考参数无效"}
        if len(prompt) > MAX_PROMPT_CHARS:
            return {"success": False, "content": "提示词最多4000字符，本次未提交。"}
        text = prompt.strip()
        stream_id = self._stream_id(kwargs)
        if not self.config.plugin.enabled:
            return {"success": False, "content": "生图插件未启用"}
        if not text:
            return {"success": False, "content": "提示词是空的"}
        if not stream_id:
            return {"success": False, "content": "找不到当前聊天"}
        active = self._active_task(stream_id)
        if active is not None:
            return {"success": False, "content": "上一张还没画完。\n" + active.summary()}
        images = None
        if use_self_reference:
            reference = self.config.self_portrait
            settings = (reference.enabled, reference.reference_path)
            if not reference.enabled or not reference.reference_path:
                return {"success": False, "content": "未启用或配置自画像参考图，未提交生图；请管理员配置参考图。"}
            try:
                images = [await image_work(read_reference, reference.reference_path)]
            except Exception:
                return {"success": False, "content": "自画像参考图无法读取、已变化或格式/大小无效，未生成，不退回纯文生图。请管理员检查参考文件。"}
            if not self._current(generation) or settings != (self.config.self_portrait.enabled, self.config.self_portrait.reference_path):
                return {"success": False, "content": "参考图配置在读取期间已更改，本次未提交。"}
            active = self._active_task(stream_id)
            if active is not None:
                return {"success": False, "content": "上一张还没画完。\n" + active.summary()}
            text = portrait_prompt(text)
        extra = {"aspect_ratio": ratio} if ratio else {}
        task = self._start_task(stream_id, text, str(kwargs.get("user_id") or ""), images, **extra) if images else self._start_task(stream_id, text, str(kwargs.get("user_id") or ""), **extra)
        task.display_prompt = prompt.strip()
        return {
            "success": True,
            "content": ("已附角色参考图，按图生图提交。" if images else "") + "提交成功，不代表已经生成成功。完成后直接发图；以下仅为提交时快照，后续以插件实时状态或 neko_draw_status 为准。\n" + task.summary(),
        }

    @Tool(
        "neko_edit_image",
        trusted_context="neko.tools.context.v1",
        brief_description="改图/图生图：修改聊天图片或GIF；生成全新图片用neko_draw。",
        detailed_description=(
            "用于换背景、换装、重绘或参考原图。GIF保留动画，其他动图格式取首帧。"
            "自参考编辑最多3张聊天源图加1张角色参考；源图或参考不可用均失败，不降级。"
            "找不到源图请用户补发，不用文生图冒充。涉及机器人外貌时按use_self_reference说明处理。"
        ),
        parameters=[
            ToolParameterInfo(name="prompt", param_type=ToolParamType.STRING, description="修改要求和保留内容；GIF保留动画，其他动图格式取首帧。GIF编辑自动补充“高分辨率高帧”，不保证效果", required=True),
            ToolParameterInfo(name="source_message_id", param_type=ToolParamType.STRING, description="当前聊天中图片、表情包或引用消息的真实ID，也支持短消息编号、工具媒体索引；不能编造。预览可能压缩，不保证原始文件；找不到源图请用户补发，不用文生图冒充。", required=True),
            ToolParameterInfo(name="source_image_index", param_type=ToolParamType.INTEGER, description="指定某张普通图片，从0开始；省略时保留多图/引用。不用于emoji表情包", required=False),
            ToolParameterInfo(name="use_self_reference", param_type=ToolParamType.BOOLEAN, description="修改涉及机器人角色外貌时按当前自画像配置的说明使用；不填本地路径。", required=False),
            ToolParameterInfo(name="aspect_ratio", param_type=ToolParamType.STRING,
                              description=EDIT_RATIO_HINT, enum_values=[*ASPECT_RATIOS, "source"], required=False),
        ],
    )
    async def edit_image(self, prompt: str = "", source_message_id: str = "", source_image_index: int | None = None, use_self_reference: bool = False, aspect_ratio: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        identity = self._tool_identity(kwargs.get("_host_context"))
        if identity is None:
            return {"success": False, "content": "可信聊天身份缺失或矛盾，本次未提交；请管理员核对Host可信上下文与会话路由，不要自行填写身份重试。"}
        return await self._edit_image(prompt, source_message_id, source_image_index, use_self_reference, aspect_ratio, **identity)

    @admitted
    async def _edit_image(self, prompt: str = "", source_message_id: str = "", source_image_index: int | None = None, use_self_reference: bool = False, aspect_ratio: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        generation = self._generation_id()
        try:
            ratio = normalize_aspect_ratio(aspect_ratio)
        except ValueError as exc:
            return {"success": False, "content": str(exc) + "，未提交。"}
        if not isinstance(prompt, str) or not isinstance(source_message_id, str) or type(use_self_reference) is not bool:
            return {"success": False, "content": "修改要求、源图编号或自参考参数无效"}
        stream_id = self._stream_id(kwargs)
        if len(prompt) > MAX_PROMPT_CHARS:
            return {"success": False, "content": "提示词最多4000字符，本次未提交。"}
        if not self.config.plugin.enabled or not stream_id or not prompt.strip():
            return {"success": False, "content": "插件未启用、聊天缺失或修改要求为空"}
        if self._active_task(stream_id):
            return {"success": False, "content": "上一张还没画完。\n" + self._status_text(stream_id)}
        try:
            images = await resolve_edit_images(self.ctx, stream_id, source_message_id.strip(), kwargs.get("message"), self._image_cache, index=source_image_index)
        except Exception as exc:
            return {"success": False, "content": "无法取得聊天源图：" + error_excerpt(self._redact(str(exc)), 300)}
        if not self._current(generation):
            return {"success": False, "content": "插件生命周期已更改，本次未提交。"}
        source_count = len(images)
        if not images:
            return {"success": False, "content": "没有取得聊天源图，未提交；角色参考不能代替待修改源图。"}
        ratio_origin = "显式参数"
        canvas = None
        if ratio in (None, "source"):
            try:
                canvas = await image_work(source_canvas, images[0])
                ratio = nearest_ratio(canvas[0] / canvas[1])
            except Exception:
                return {"success": False, "content": "无法读取主源图画布尺寸，本次未提交。"}
            ratio_origin = f"主源图画布{canvas[0]}×{canvas[1]}匹配（最接近支持比例，不保证原像素尺寸）"
            if not self._current(generation):
                return {"success": False, "content": "插件生命周期已更改，本次未提交。"}
        text = prompt.strip()
        if any(image.startswith((b'GIF87a', b'GIF89a')) for image in images) and '高分辨率高帧' not in text:
            text += '\n高分辨率高帧'
        if use_self_reference:
            if source_count > 3:
                return {"success": False, "content": "自参考编辑最多3张聊天源图加1张角色参考；请用source_image_index明确选择源图，未自动丢弃图片。"}
            settings = (self.config.self_portrait.enabled, self.config.self_portrait.reference_path)
            if not settings[0] or not settings[1]:
                return {"success": False, "content": "未启用或配置角色参考图，本次未提交；不会忽略自参考选项。"}
            try:
                reference = await image_work(read_reference, settings[1])
            except Exception:
                return {"success": False, "content": "角色参考图无法读取或无效，本次未提交，不降级为普通编辑；请管理员检查参考图。"}
            if not self._current(generation) or settings != (self.config.self_portrait.enabled, self.config.self_portrait.reference_path):
                return {"success": False, "content": "角色参考配置已更改，本次未提交。"}
            images = [*images, reference]
            text = portrait_edit_prompt(text, source_count)
        if not self.config.plugin.enabled:
            return {"success": False, "content": "插件已禁用，本次未提交。"}
        if self._active_task(stream_id):
            return {"success": False, "content": "上一张还没画完。\n" + self._status_text(stream_id)}
        task = self._start_task(stream_id, text, str(kwargs.get("user_id") or ""), images,
                                aspect_ratio=ratio, aspect_ratio_origin=ratio_origin, source_canvas=canvas,
                                aspect_ratio_automatic=aspect_ratio is None)
        task.display_prompt = prompt.strip()
        notice = f"已读取 {source_count} 张聊天源图（平台或上下文预览可能压缩，不保证原始文件）" + ("，并在最后附1张角色外貌参考" if use_self_reference else "")
        return {"success": True, "content": notice + "并提交图生图，不代表已经生成成功。完成后直接发图；以下仅为提交时快照，后续以插件实时状态或 neko_draw_status 为准。\n" + task.summary()}

    @Tool(
        "neko_draw_status",
        trusted_context="neko.tools.context.v1",
        brief_description="查询当前聊天最近一次画图/生图的状态、进度或失败原因；不开始新绘图。",
        detailed_description="失败原因仅提供最多180字符的错误摘录；长说明标记截断，不提供完整接口原文。",
        parameters=[],
    )
    async def draw_status(self, **kwargs: Any) -> Dict[str, Any]:
        identity = self._tool_identity(kwargs.get("_host_context"))
        if identity is None:
            return {"success": False, "content": "可信聊天身份缺失或矛盾，未查询；请管理员核对Host可信上下文与会话路由，不要自行填写身份重试。"}
        return {"success": True, "content": self._status_text(identity["stream_id"])}

    @staticmethod
    def _tool_identity(context):
        # Host must overwrite this envelope for every opt-in invocation. Ordinary
        # model kwargs (including message/stream_id) never supply identity or bytes.
        if not isinstance(context, dict) or context.get("contract") != "neko.tools.context.v1":
            return None
        session = context.get("session_id")
        if (not isinstance(session, str) or not 0 < len(session) <= 256 or session != session.strip()
                or any(ord(c) < 32 for c in session) or context.get("stream_id") != session):
            return None
        # Tools are session-scoped, not command-whitelist scoped. Normal and
        # Focus Host contexts have no anchor and therefore may have no account.
        keys = ("session_id", "stream_id", "platform", "account_id", "group_id", "user_id")
        return {key: context.get(key, "") for key in keys}

    def _command_identity(self, stream_id, user_id, kwargs):
        # Command args are Host-produced, not LLM arguments. Cross-check against
        # the actual inbound message, which also supplies the receiving account.
        message = kwargs.get("message")
        if not isinstance(message, dict):
            return None
        info = message.get("message_info")
        if not isinstance(info, dict) or not isinstance(info.get("user_info"), dict):
            return None
        group = info.get("group_info") or {}
        if not isinstance(group, dict):
            return None
        context = dict(contract="neko.tools.context.v1", session_id=message.get("session_id"),
                       stream_id=stream_id, platform=message.get("platform"), account_id=message.get("account_id"),
                       user_id=info["user_info"].get("user_id"), group_id=group.get("group_id", ""))
        identity = self._tool_identity(context)
        if identity is None:
            return None
        if any(not isinstance(value, str) or value != value.strip() or len(value) > 256
               or any(ord(c) < 32 for c in value) for value in identity.values()):
            return None
        if (identity["platform"] not in ("qq", "telegram") or not identity["account_id"]
                or not user_id or identity["user_id"] != user_id
                or identity["platform"] != kwargs.get("platform")
                or identity["group_id"] != kwargs.get("group_id", "")):
            return None
        for key in ("session_id", "account_id", "chat_id"):
            if key in kwargs and kwargs[key] != identity["session_id" if key == "chat_id" else key]:
                return None
        return identity

    def _allowed(self, user_id: str, platform: str) -> bool:
        if platform == "qq":
            values = self.config.command.allowed_user_ids
        elif platform == "telegram":
            values = self.config.command.allowed_telegram_user_ids
            if re.fullmatch(r"[1-9][0-9]{0,19}", user_id) is None:
                return False
        else:
            return False
        allowed = {str(item).strip() for item in values if str(item).strip()}
        return bool(user_id) and user_id in allowed

    @Command(
        "neko_draw_cmd",
        description="手动生图或查看状态",
        pattern=r"^/(?:draw|生图|画图)(?:\s+(?P<payload>[\s\S]+))?$",
    )
    async def cmd_draw(
        self,
        stream_id: str = "",
        user_id: str = "",
        matched_groups: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str, bool]:
        identity = self._command_identity(stream_id, user_id, kwargs)
        if identity is None:
            return False, "可信聊天身份缺失或矛盾；请管理员核对入站身份与会话路由，不要自行填写身份重试", True
        sender = identity["user_id"]
        if not self._allowed(sender, identity["platform"]):
            self.ctx.logger.info("拒绝 /draw：用户 %s 不在白名单", sender or "未知")
            return False, "生图命令未对你开放", True
        if not stream_id:
            return False, "找不到当前聊天", True
        payload = str((matched_groups or {}).get("payload") or "").strip()
        if not payload or payload in {"status", "状态"}:
            return True, self._status_text(stream_id), True
        parts = payload.split(maxsplit=1)
        if parts[0].lower() in {"edit", "图生图", "改图"}:
            if len(parts) < 2:
                return False, "请引用原图，发送 /draw edit <修改要求>", True
            result = await self._edit_image(prompt=parts[1], message=kwargs.get("message"), **identity)
            return bool(result["success"]), str(result["content"]), True
        if not self.config.plugin.enabled:
            return False, "生图插件未启用", True
        active = self._active_task(stream_id)
        if active is not None:
            return False, "上一张还没画完。\n" + active.summary(), True
        result = await self._draw(prompt=payload, **identity)
        return bool(result['success']), result['content'], True


def create_plugin() -> NekoDraw:
    return NekoDraw()
