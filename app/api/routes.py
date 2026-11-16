"""Versioned REST API: chat, telemetry, presets and health."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response

from app import __version__
from app.api.dependencies import ContainerDep
from app.api.middleware import get_request_id
from app.api.presets import PRESETS
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    Preset,
    TelemetrySnapshot,
)
from app.providers.base import ProviderError

router = APIRouter(prefix="/api/v1", tags=["llm-guard"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a prompt through the firewall to the LLM",
    responses={
        413: {"model": ErrorResponse, "description": "Prompt exceeds app.max_input_chars"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
async def chat(payload: ChatRequest, request: Request, container: ContainerDep) -> ChatResponse:
    """Run the prompt through the input layers, the LLM and the output guard.

    The response carries the complete security report: total risk score,
    per-layer scores and findings, triggered rules and timings.
    """
    limit = container.settings.app.max_input_chars
    if len(payload.message) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"message exceeds the {limit}-character limit",
        )
    try:
        return await container.chat_service.handle(
            message=payload.message,
            firewall_enabled=payload.firewall_enabled,
            request_id=get_request_id(request) or "unknown",
        )
    except ProviderError as exc:
        raise HTTPException(
            status_code=503,
            detail="the LLM provider is unavailable",
        ) from exc


@router.get(
    "/telemetry",
    response_model=TelemetrySnapshot,
    summary="Aggregated security metrics and the latest security events",
)
async def telemetry(container: ContainerDep) -> TelemetrySnapshot:
    return container.telemetry.snapshot()


@router.delete(
    "/telemetry",
    status_code=204,
    summary="Reset telemetry (demo mode only)",
    responses={403: {"model": ErrorResponse, "description": "Not available outside demo mode"}},
)
async def reset_telemetry(container: ContainerDep) -> Response:
    if not container.settings.app.demo_mode:
        raise HTTPException(
            status_code=403,
            detail="telemetry reset is only available in demo mode",
        )
    container.telemetry.reset()
    return Response(status_code=204)


@router.get(
    "/presets",
    response_model=list[Preset],
    summary="One-click playground scenarios (benign prompts and attacks)",
)
async def presets() -> list[Preset]:
    return list(PRESETS)


@router.get("/health", response_model=HealthResponse, summary="Liveness and component status")
async def health(container: ContainerDep) -> HealthResponse:
    metadata = container.firewall.model_metadata
    await container.provider.is_available()  # refreshes the (cached) fallback status
    return HealthResponse(
        version=__version__,
        demo_mode=container.settings.app.demo_mode,
        provider=container.provider.describe(),
        fallback_active=container.provider.fallback_active,
        rules_loaded=container.firewall.rules_loaded,
        flag_threshold=container.settings.policy.thresholds.flag,
        block_threshold=container.settings.policy.thresholds.block,
        max_input_chars=container.settings.app.max_input_chars,
        ml_model=(
            ModelInfo(
                algorithm=metadata.algorithm,
                samples=metadata.samples,
                cv_f1=metadata.cv_f1,
                cv_accuracy=metadata.cv_accuracy,
                trained_at=metadata.trained_at,
            )
            if metadata is not None
            else None
        ),
    )
