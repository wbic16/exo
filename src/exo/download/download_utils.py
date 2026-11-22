"""Model-file helpers for the rest of exo: finding a model on disk, fetching a
single file (model cards), and an HTTP session for other callers."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import aiohttp
from anyio import to_thread
from pydantic import ValidationError

from exo.download import store, transfer
from exo.download.hub import Hub, NotFoundError, new_session, proxy_for
from exo.download.store import Manifest
from exo.download.transfer import DownloadSettings, Locks, Part, Progress
from exo.shared.models.model_cards import ConfigData, ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


def resolve_existing_model(
    model_id: ModelId, card: ModelCard | None = None
) -> Path | None:
    """The dir of a complete copy of the model (read-only dirs first), if any."""
    return store.find_complete(model_id, card)


def is_read_only_model_dir(model_dir: Path) -> bool:
    return store.is_read_only(model_dir)


def build_model_path(model_id: ModelId) -> Path:
    found = store.find_complete(model_id)
    return found if found is not None else store.default_dir() / model_id.normalize()


def build_local_model_card(model_id: ModelId) -> ModelCard | None:
    """A ModelCard built from a complete local copy of the model, if one is
    on disk: config.json supplies the architecture fields and the safetensors
    files on disk supply the storage size. Lets exo place and serve a model
    that exists only locally (never published to a hub) with no network
    access. Returns None when there is no complete local copy, or its
    config.json cannot be parsed, or it holds no safetensors."""
    found = store.find_complete(model_id)
    if found is None:
        return None
    try:
        config = json.loads((found / "config.json").read_text())
        config_data = ConfigData.model_validate(config)
    except (OSError, ValueError, ValidationError):
        return None
    storage_bytes = sum(
        p.stat().st_size for p in found.glob("*.safetensors") if p.is_file()
    )
    if storage_bytes == 0:
        return None
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(storage_bytes),
        n_layers=config_data.layer_count,
        hidden_size=config_data.hidden_size or 0,
        supports_tensor=config_data.supports_tensor,
        num_key_value_heads=config_data.num_key_value_heads,
        context_length=config_data.max_position_embeddings,
        tasks=[ModelTask.TextGeneration],
        trust_remote_code=False,
        is_custom=True,
        vision=config_data.vision,
        backends=list(
            Backend
        ),  # all backends, as in fetch_from_hf: placement gates decide
        model_type=config.get("model_type"),
    )


async def resolve_model_dir(model_id: ModelId) -> Path:
    """The model's dir (a complete copy, else the default one), created if needed."""
    target = await to_thread.run_sync(build_model_path, model_id)
    await to_thread.run_sync(lambda: target.mkdir(parents=True, exist_ok=True))
    return target


def create_http_session(
    auto_decompress: bool = False,
    timeout_profile: Literal["short", "long"] = "long",
) -> aiohttp.ClientSession:
    # one proxy for the session's requests, as before (any host)
    proxy = proxy_for("https://")
    if timeout_profile == "short":
        return new_session(
            connect_timeout=10,
            idle_timeout=30,
            total_timeout=30,
            auto_decompress=auto_decompress,
            proxy=proxy,
        )
    return new_session(
        connect_timeout=60,
        idle_timeout=60,
        auto_decompress=auto_decompress,
        proxy=proxy,
    )


_TRIES = 3


async def download_file_with_retry(
    model_id: ModelId,
    revision: str,
    path: str,
    target_dir: Path,
    on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
) -> Path:
    """Fetch one file of the repo into ``target_dir`` unless it is already there,
    the same way a model download does (resumed, hash-checked, renamed into place).

    Uses the commit a download of this model pinned, if any, else ``revision``.
    Raises ``FileNotFoundError`` when the Hub doesn't have it.
    """
    target = target_dir / path
    if await to_thread.run_sync(target.exists):
        return target
    settings = DownloadSettings()
    progress = Progress()
    async with Hub.connect(
        connect_timeout=settings.connect_timeout,
        idle_timeout=settings.idle_timeout,
        metadata_timeout=settings.metadata_timeout,
    ) as hub:

        async def locate() -> Manifest:
            pinned = await to_thread.run_sync(store.read_manifest, target_dir)
            if pinned is not None:
                return pinned
            commit = await hub.resolve_commit(model_id, revision)
            return Manifest(
                repo=str(model_id),
                commit=commit,
                files=await hub.list_files(model_id, commit),
            )

        try:
            # it answers an API call: a few tries, as before, not a long wait
            manifest = await transfer.retrying(
                locate, progress, settings, max_tries=_TRIES
            )
            file = next((f for f in manifest.files if f.path == path), None)
            if file is None:
                raise FileNotFoundError(f"{path} is not in {model_id}.")
            part = Part(model_id, target_dir, manifest, check_index=False)
            await transfer.fetch_file(
                hub, part, file, progress, settings, Locks(), max_tries=_TRIES
            )
        except NotFoundError as error:
            raise FileNotFoundError(error.message) from error
    on_progress(file.size, file.size, True)
    return target
