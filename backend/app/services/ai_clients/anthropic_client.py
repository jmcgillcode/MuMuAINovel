"""Anthropic 客户端"""
from typing import Any, AsyncGenerator, Dict, Optional
import json

from app.logger import get_logger
from .base_client import BaseAIClient

logger = get_logger(__name__)


class AnthropicClient(BaseAIClient):
    """Anthropic API 客户端"""

    def _build_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    def _build_payload(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if stream:
            payload["stream"] = True
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = tools
            if tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
        return payload

    async def chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._build_payload(messages, model, temperature, max_tokens, system_prompt, tools, tool_choice)

        data = await self._request_with_retry("POST", "/v1/messages", payload)

        tool_calls = []
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {"name": block.get("name"), "arguments": block.get("input")},
                })
            elif block.get("type") == "text":
                content += block.get("text", "")

        return {
            "content": content,
            "tool_calls": tool_calls if tool_calls else None,
            "finish_reason": data.get("stop_reason"),
        }

    async def chat_completion_stream(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式生成"""
        payload = self._build_payload(messages, model, temperature, max_tokens, system_prompt, tools, tool_choice, stream=True)

        tool_calls = []

        try:
            async with await self._request_with_retry("POST", "/v1/messages", payload, stream=True) as response:
                response.raise_for_status()
                try:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            event = json.loads(data_str)
                            event_type = event.get("type")

                            if event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield {"content": delta.get("text", "")}

                            elif event_type == "message_delta":
                                if event.get("delta", {}).get("stop_reason"):
                                    if tool_calls:
                                        yield {"tool_calls": tool_calls}
                                    yield {"done": True, "finish_reason": event["delta"]["stop_reason"]}

                        except json.JSONDecodeError:
                            continue

                except GeneratorExit:
                    logger.debug("Anthropic 流式响应生成器被关闭")
                    raise
                except Exception as iter_error:
                    logger.error(f"Anthropic 流式响应迭代出错: {str(iter_error)}")
                    raise
        except GeneratorExit:
            raise
        except Exception as e:
            logger.error(f"Anthropic 流式请求出错: {str(e)}")
            raise
