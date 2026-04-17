"""Configuration-driven flow builder and runner for MessyText pipelines.

This module turns a validated :class:`src.flow_loader.FlowSchema` into a
ready-to-execute pipeline on top of the existing processors in
:mod:`src.processors`. It contains no new processing logic: every LLM call,
prompt construction, and JSON schema handling is delegated to the classes
already defined in :mod:`src.processors`. The role of this module is strictly
orchestration — wiring classes together according to the flow YAML and
driving their loops.

Contents and relationships
--------------------------

- :func:`build_flow` — public entry point. Loads and validates the flow
  YAML, loads ``.env`` into ``os.environ``, loads the taxonomy and prompts
  JSON files, constructs one :class:`openai.AsyncOpenAI` client per LLM
  resource, and returns a :class:`FlowRunner`.
- :class:`FlowRunner` — owns the execution loop. :meth:`FlowRunner.run`
  loads the input CSV, dispatches each step in order, and writes the output
  CSVs declared in :class:`src.flow_loader.OutputConfig`.
- :class:`_ConversationSummaryPlan` — internal helper that pairs
  ``conversation_summary_first`` with ``conversation_summary_update`` so
  the two schema steps are executed as one sequential per-entity loop,
  matching :mod:`scripts.run_summary_conversation`.
- :func:`_load_dotenv_into_environ` — robust ``.env`` loader that uses
  ``python-dotenv`` when available and falls back to a manual
  ``KEY=VALUE`` parser when the dependency is not installed. The project
  root is resolved from ``__file__``.
- :func:`_resolve_resource_credentials` — normalises every resource so it
  has a concrete ``api_base`` and a concrete ``api_key`` string after
  reading ``api_key_env`` from the environment.
- :func:`_build_processor_runtime_config` — shapes the nested dict that
  :class:`src.processors.AsyncMessyTextProcessor` expects at construction
  time, combining the resource's generation parameters with the shared
  prompts payload.

How the rest of the system uses this module
-------------------------------------------

:mod:`scripts.run_custom_flow` is the only intended caller. It sets a
module-level ``flow_config`` variable, calls :func:`build_flow`, and runs
the returned :class:`FlowRunner`. No CLI flag parsing, no global state.

Invariants enforced by this module
----------------------------------

- Every LLM resource ends up with a non-empty ``api_key`` before any
  client is constructed. ``api_key_env`` is resolved against
  ``os.environ`` and a missing variable fails fast with a clear error.
- Every :class:`src.flow_loader.StepConfig` referenced at runtime
  corresponds to an implemented dispatcher. Step types declared in the
  schema but not yet implemented raise :class:`NotImplementedError` with
  an actionable message pointing at the existing script that still covers
  that flow.
- Output directories are created before any CSV is written.
- Runner-owned state (``running_summary``,
  :class:`src.processors.MessyTextConversationState`) is managed in
  :class:`FlowRunner`, not in the processors.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as tqdm_async

from src.flow_loader import (
    FlowSchema,
    LLMResource,
    LoggingConfig,
    PROVIDER_DEFAULT_API_BASE,
    StepConfig,
)
from src.processors import (
    AsyncLabelExtractor,
    AsyncMessyTextConversationTurnProcessor,
    AsyncMessyTextProcessor,
    AsyncTextConversationOrchestrator,
    AsyncTextLabelsSummaryProcessor,
    MessyTextConversationState,
    ProcessorResult,
)
from src.recorders import (
    flatten_spans_from_state,
    serialize_result_entry,
    serialize_state_entry,
    write_results,
    write_spans,
    write_states,
)
from src.utils import is_informative_summary, setup_logger


_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
"""Absolute path of the project root directory (the parent of ``src/``)."""


def _load_dotenv_into_environ(project_root: Path) -> None:
    """Load ``project_root/.env`` into :data:`os.environ`.

    Uses :mod:`dotenv` when importable; otherwise falls back to a minimal
    ``KEY=VALUE`` parser that handles blank lines, ``#`` comments, and
    values optionally wrapped in matching single or double quotes. Missing
    ``.env`` files are silently ignored so the rest of the flow still runs
    when every resource uses a literal ``api_key``.

    Args:
        project_root (Path): Directory that contains the ``.env`` file.

    Returns:
        None: This function mutates :data:`os.environ` in place.
    """
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass

    with env_path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            if "=" not in stripped_line:
                continue
            key_part, _, value_part = stripped_line.partition("=")
            key_name = key_part.strip()
            raw_value = value_part.strip()
            if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {"'", '"'}:
                raw_value = raw_value[1:-1]
            os.environ.setdefault(key_name, raw_value)


def _resolve_resource_credentials(resource: LLMResource) -> LLMResource:
    """Fill in ``api_base`` and ``api_key`` for a validated resource.

    ``api_base`` falls back to
    :data:`src.flow_loader.PROVIDER_DEFAULT_API_BASE` when omitted.
    ``api_key`` is resolved from :attr:`LLMResource.api_key_env` against
    :data:`os.environ`. For ``local_vllm`` the convention from the existing
    codebase is to pass the literal string ``"dummy"`` when no explicit
    key is provided.

    Args:
        resource (LLMResource): The validated resource whose credentials
            still need to be materialised.

    Returns:
        LLMResource: A new resource instance with ``api_base`` and
        ``api_key`` populated and ``api_key_env`` cleared.

    Raises:
        ValueError: If ``api_key_env`` is declared but the named
            environment variable is missing, or if a hosted provider ends
            up with no usable API key.
    """
    resolved_api_base = resource.api_base or PROVIDER_DEFAULT_API_BASE[resource.provider]

    if resource.api_key is not None:
        resolved_api_key: str = resource.api_key
    elif resource.api_key_env is not None:
        resolved_api_key_opt = os.environ.get(resource.api_key_env)
        if not resolved_api_key_opt:
            raise ValueError(
                f"Resource {resource.id!r} declares api_key_env="
                f"{resource.api_key_env!r}, but that environment variable "
                "is not set. Populate it in the project-root .env file or "
                "export it before running."
            )
        resolved_api_key = resolved_api_key_opt
    elif resource.provider == "local_vllm":
        resolved_api_key = "dummy"
    else:
        raise ValueError(
            f"Resource {resource.id!r} (provider={resource.provider!r}) has "
            "neither api_key nor api_key_env set. Hosted providers require "
            "api_key_env pointing at a variable in the .env file."
        )

    return resource.model_copy(
        update={
            "api_base": resolved_api_base,
            "api_key": resolved_api_key,
            "api_key_env": None,
        }
    )


def _build_async_client(resource: LLMResource, max_retries: int) -> AsyncOpenAI:
    """Construct an :class:`openai.AsyncOpenAI` client for a resolved resource.

    Args:
        resource (LLMResource): A resource that has been passed through
            :func:`_resolve_resource_credentials` so ``api_base`` and
            ``api_key`` are non-None.
        max_retries (int): Maximum retry count forwarded to the client.

    Returns:
        AsyncOpenAI: A configured asynchronous client instance.

    Raises:
        ValueError: If ``api_base`` or ``api_key`` is missing on the
            resource after resolution.
    """
    if resource.api_base is None or resource.api_key is None:
        raise ValueError(
            f"Resource {resource.id!r} is missing api_base or api_key after "
            "resolution. Call _resolve_resource_credentials first."
        )
    return AsyncOpenAI(
        base_url=resource.api_base,
        api_key=resource.api_key,
        max_retries=max_retries,
    )


def _build_processor_runtime_config(
    resource: LLMResource,
    prompts_payload: Dict[str, Any],
    logging_config: LoggingConfig,
) -> Dict[str, Any]:
    """Shape the runtime config dict consumed by processor constructors.

    :class:`src.processors.AsyncMessyTextProcessor` and its siblings read a
    nested dict at construction time. This helper assembles that dict from
    the resource's generation parameters and the loaded prompts payload so
    the runner does not hand-roll the same shape in every dispatcher.

    Args:
        resource (LLMResource): The resolved resource whose model name and
            generation parameters are copied into the config.
        prompts_payload (Dict[str, Any]): Parsed ``config/prompts.json``
            content. Surfaced under the ``prompts`` key.
        logging_config (LoggingConfig): The flow's logging configuration,
            surfaced under the ``logging`` key so processors can honour
            :attr:`LoggingConfig.log_response`.

    Returns:
        Dict[str, Any]: A dict with keys ``model``, ``processing``,
        ``prompts``, and ``logging``, shaped to match what
        :class:`src.processors.AsyncMessyTextProcessor` expects.
    """
    return {
        "model": {"name": resource.model},
        "processing": {
            "temperature": resource.temperature,
            "max_tokens_summary": resource.max_tokens_summary,
            "max_tokens_classification": resource.max_tokens_classification,
        },
        "prompts": prompts_payload,
        "logging": logging_config.model_dump(),
    }


def _load_json_file(json_path: Path) -> Dict[str, Any]:
    """Read a JSON file and return its parsed contents.

    Args:
        json_path (Path): Path to the JSON file on disk.

    Returns:
        Dict[str, Any]: The parsed JSON object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a JSON object.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object at {json_path}, got {type(payload).__name__}"
        )
    return payload


class _ConversationSummaryPlan:
    """Holder that marks whether the conversation summary pair has been executed.

    The two schema steps ``conversation_summary_first`` and
    ``conversation_summary_update`` are executed as a single sequential
    per-entity loop (matching :mod:`scripts.run_summary_conversation`).
    This helper records that the loop has already run so the second of
    the two steps does not trigger a duplicate pass.

    Attributes:
        executed (bool): ``True`` once the per-entity loop has run.
    """

    def __init__(self) -> None:
        """Initialise the plan with ``executed = False``."""
        self.executed: bool = False


class _LabelPlan:
    """Holder that marks whether the label extraction + summary pair ran.

    ``label_extraction`` and ``label_summary`` are tightly coupled: the
    runner executes both in a single pass over entities. This flag
    prevents the second step from triggering a duplicate pass.

    Attributes:
        executed (bool): ``True`` once the combined loop has run.
    """

    def __init__(self) -> None:
        """Initialise the plan with ``executed = False``."""
        self.executed: bool = False


class _Checkpoint:
    """Entity-level checkpoint manager for resumable flows.

    After each entity completes all pipeline steps the runner calls
    :meth:`mark_completed`. On resume, :meth:`completed_entity_ids`
    returns the set of entity ids that already succeeded so they can be
    skipped.

    Attributes:
        checkpoint_dir (Path): Directory where checkpoint files are
            stored.
        completed_file (Path): JSON file listing completed entity ids.
    """

    def __init__(self, output_dir: Path, flow_name: str) -> None:
        """Initialise the checkpoint under ``output_dir/.checkpoint/``.

        Args:
            output_dir (Path): Parent of the summary CSV.
            flow_name (str): Used in log messages.
        """
        self.checkpoint_dir = output_dir / ".checkpoint"
        self.completed_file = self.checkpoint_dir / "completed_entities.json"
        self._flow_name = flow_name

    def completed_entity_ids(self) -> Set[str]:
        """Load previously completed entity ids from disk.

        Returns:
            Set[str]: Entity ids (stringified) that already succeeded.
        """
        if not self.completed_file.exists():
            return set()
        with self.completed_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed", []))

    def mark_completed(self, entity_id: Any) -> None:
        """Append one entity id to the checkpoint file.

        Args:
            entity_id (Any): The entity id to record.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        completed = self.completed_entity_ids()
        completed.add(str(entity_id))
        with self.completed_file.open("w", encoding="utf-8") as f:
            json.dump({"completed": sorted(completed)}, f, indent=2)

    def clear(self) -> None:
        """Remove the checkpoint file to start fresh."""
        if self.completed_file.exists():
            self.completed_file.unlink()


class FlowRunner:
    """Execute a validated :class:`src.flow_loader.FlowSchema` end-to-end.

    The runner owns the input DataFrame, all output writers, the logger,
    and the asyncio semaphore used to cap concurrent LLM calls. It does not
    own any LLM-call logic: every per-turn call goes through the
    appropriate processor class from :mod:`src.processors`.

    Supported step types (all entries in
    :data:`src.flow_loader.STEP_TYPES`):

    - ``conversation_summary_first`` / ``conversation_summary_update``
    - ``single_summary``
    - ``classification``
    - ``label_extraction`` / ``label_summary`` (hybrid and full_async)
    - ``evaluation``

    Attributes:
        schema (FlowSchema): The validated flow definition.
        resolved_resources (Dict[str, LLMResource]): Resolved resources
            keyed by ``id``, each with ``api_base`` and ``api_key``
            populated.
        clients_by_id (Dict[str, AsyncOpenAI]): One AsyncOpenAI client per
            resource id.
        processor_configs (Dict[str, Dict[str, Any]]): Shaped processor
            runtime config per resource id.
        taxonomy (Dict[str, Any]): Parsed taxonomy JSON.
        prompts (Dict[str, Any]): Parsed prompts JSON.
        logger (logging.Logger): Logger configured for this flow.

    Methods:
        run: Execute the flow synchronously by wrapping
            :meth:`run_async` in :func:`asyncio.run`.
        run_async: Execute every step of the flow on the current event
            loop.
    """

    def __init__(
        self,
        schema: FlowSchema,
        resolved_resources: Dict[str, LLMResource],
        clients_by_id: Dict[str, AsyncOpenAI],
        processor_configs: Dict[str, Dict[str, Any]],
        taxonomy: Dict[str, Any],
        prompts: Dict[str, Any],
        logger: logging.Logger,
        resume: bool = False,
    ) -> None:
        """Store the wired objects produced by :func:`build_flow`.

        Args:
            schema (FlowSchema): The validated flow definition.
            resolved_resources (Dict[str, LLMResource]): Resolved resources
                keyed by ``id``.
            clients_by_id (Dict[str, AsyncOpenAI]): One AsyncOpenAI client
                per resource id.
            processor_configs (Dict[str, Dict[str, Any]]): Shaped processor
                runtime config per resource id.
            taxonomy (Dict[str, Any]): Parsed taxonomy JSON.
            prompts (Dict[str, Any]): Parsed prompts JSON.
            logger (logging.Logger): Logger configured for this flow.
            resume (bool): When ``True`` the runner loads entity-level
                checkpoints and skips already-completed entities.

        Returns:
            None.
        """
        self.schema = schema
        self.resolved_resources = resolved_resources
        self.clients_by_id = clients_by_id
        self.processor_configs = processor_configs
        self.taxonomy = taxonomy
        self.prompts = prompts
        self.logger = logger
        self.resume = resume

        output_dir = Path(schema.flow.output.summary_csv).parent
        self._checkpoint = _Checkpoint(output_dir, schema.flow.name)

    def run(self) -> None:
        """Execute the flow synchronously.

        Wraps :meth:`run_async` in :func:`asyncio.run`. Suitable as the
        top-level call from a script.

        Returns:
            None.
        """
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        """Execute every step of the flow on the current event loop.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the input CSV does not exist.
            KeyError: If a role column is missing from the CSV.
            NotImplementedError: If the flow declares an unimplemented step.
        """
        flow = self.schema.flow
        input_df = self._load_input_csv()
        model_name = (
            self.resolved_resources["default"].model
            if "default" in self.resolved_resources
            else next(iter(self.resolved_resources.values())).model
        )

        column_roles = flow.data.column_roles
        has_entity_grouping = any(
            s.type
            in {
                "conversation_summary_first",
                "conversation_summary_update",
                "label_extraction",
                "label_summary",
            }
            or s.unit in {"document", "entity"}
            for s in flow.steps
        )

        if flow.processing_limit is not None:
            if has_entity_grouping:
                entity_ids = input_df[column_roles.entity_id].unique()[: flow.processing_limit]
                input_df = input_df[input_df[column_roles.entity_id].isin(entity_ids)]
            else:
                input_df = input_df.head(flow.processing_limit)
            self.logger.info(
                "Processing limit applied: %d %s",
                len(input_df[column_roles.entity_id].unique()) if has_entity_grouping else len(input_df),
                "entities" if has_entity_grouping else "rows",
            )

        conversation_plan = _ConversationSummaryPlan()
        label_plan = _LabelPlan()
        entity_states: List[Tuple[Any, MessyTextConversationState]] = []
        processed_df = input_df.copy()
        if "summary_all_context" not in processed_df.columns:
            processed_df["summary_all_context"] = ""

        for step_index, step in enumerate(flow.steps):
            self.logger.info(
                "Dispatching step %d/%d: type=%s unit=%s",
                step_index + 1,
                len(flow.steps),
                step.type,
                step.unit,
            )

            if step.type in {"conversation_summary_first", "conversation_summary_update"}:
                if conversation_plan.executed:
                    continue
                processed_df, entity_states = await self._run_conversation_summary(
                    df=processed_df, step=step,
                )
                conversation_plan.executed = True

            elif step.type == "single_summary":
                processed_df = await self._run_single_summary(
                    df=processed_df, step=step,
                )

            elif step.type == "classification":
                processed_df = await self._run_classification(
                    df=processed_df, step=step,
                )

            elif step.type == "label_extraction":
                if label_plan.executed:
                    continue
                label_summary_step = next(
                    (s for s in flow.steps if s.type == "label_summary"), None,
                )
                mode = "hybrid"
                if label_summary_step is not None and label_summary_step.mode == "full_async":
                    mode = "full_async"

                if mode == "full_async":
                    processed_df, entity_states = await self._run_label_full_async(
                        df=processed_df, extraction_step=step,
                        summary_step=label_summary_step,
                    )
                else:
                    processed_df, entity_states = await self._run_label_hybrid(
                        df=processed_df, extraction_step=step,
                        summary_step=label_summary_step,
                    )
                label_plan.executed = True

            elif step.type == "label_summary":
                if label_plan.executed:
                    continue
                self.logger.warning(
                    "label_summary without a preceding label_extraction step; skipping.",
                )

            elif step.type == "evaluation":
                processed_df = await self._run_evaluation(
                    df=processed_df, step=step,
                )

            else:
                raise NotImplementedError(
                    f"Step type {step.type!r} is not recognised by the runner."
                )

        self._write_outputs(
            processed_df=processed_df,
            entity_states=entity_states,
            model_name=model_name,
        )

    def _load_input_csv(self) -> pd.DataFrame:
        """Load the input CSV and validate that every role column exists.

        Returns:
            pd.DataFrame: The input DataFrame loaded from
            :attr:`src.flow_loader.DataConfig.input_csv`.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            KeyError: If a role column from
                :attr:`src.flow_loader.ColumnRoles` is missing.
        """
        data_config = self.schema.flow.data
        csv_path = Path(data_config.input_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        input_df = pd.read_csv(csv_path, encoding="utf-8")
        required_columns = {
            data_config.column_roles.text,
            data_config.column_roles.entity_id,
            data_config.column_roles.doc_id,
            data_config.column_roles.sort_by,
        }
        missing_columns = required_columns - set(input_df.columns)
        if missing_columns:
            raise KeyError(
                f"Input CSV {csv_path} is missing role columns: "
                f"{sorted(missing_columns)}"
            )
        return input_df

    def _select_resource_id_for_step(self, step: StepConfig) -> str:
        """Resolve the LLM resource id used by a step.

        Args:
            step (StepConfig): The step whose resource id is being resolved.

        Returns:
            str: The resource id. Falls back to ``"default"`` when the
            step does not declare ``llm``.

        Raises:
            KeyError: If the resolved id is not among the loaded resources.
        """
        resource_id = step.llm or "default"
        if resource_id not in self.resolved_resources:
            raise KeyError(
                f"Step references llm={resource_id!r}, but no resource with "
                f"that id is loaded. Available ids: "
                f"{sorted(self.resolved_resources)}"
            )
        return resource_id

    async def _run_conversation_summary(
        self,
        df: pd.DataFrame,
        step: StepConfig,
    ) -> Tuple[pd.DataFrame, List[Tuple[Any, MessyTextConversationState]]]:
        """Run the conversation summary flow over every entity in ``df``.

        Mirrors :func:`scripts.run_summary_conversation._process_dataframe_conversation_async`:
        each entity's documents are processed sequentially with
        ``previous_summary`` threading, only informative turns update the
        running summary, and entities are processed concurrently up to
        :attr:`src.flow_loader.AsyncConfig.max_concurrent_rows`.

        Args:
            df (pd.DataFrame): The input DataFrame. Must contain the role
                columns declared in
                :attr:`src.flow_loader.ColumnRoles`.
            step (StepConfig): The step config. Used to select the LLM
                resource via :meth:`_select_resource_id_for_step`.

        Returns:
            Tuple[pd.DataFrame, List[Tuple[Any, MessyTextConversationState]]]:
            The DataFrame with ``summary_all_context`` populated, and the
            list of per-entity conversation states in completion order.

        Raises:
            KeyError: If a role column is missing after CSV load.
        """
        flow = self.schema.flow
        column_roles = flow.data.column_roles
        async_config = flow.async_config

        resource_id = self._select_resource_id_for_step(step)
        client = self.clients_by_id[resource_id]
        processor_config = self.processor_configs[resource_id]

        llm_semaphore: Optional[asyncio.Semaphore] = None
        if async_config.max_concurrent_llm_calls > 0:
            llm_semaphore = asyncio.Semaphore(async_config.max_concurrent_llm_calls)

        processor = AsyncMessyTextProcessor(
            client=client,
            config=processor_config,
            taxonomy=self.taxonomy,
            logger=self.logger,
            llm_semaphore=llm_semaphore,
        )
        turn_processor = AsyncMessyTextConversationTurnProcessor(processor)

        processed_df = df.copy()
        if "summary_all_context" not in processed_df.columns:
            processed_df["summary_all_context"] = ""

        entity_groups = list(processed_df.groupby(column_roles.entity_id))

        skip_ids: Set[str] = set()
        if self.resume:
            skip_ids = self._checkpoint.completed_entity_ids()
            if skip_ids:
                self.logger.info(
                    "Resuming: skipping %d already-completed entities.", len(skip_ids),
                )

        concurrency_semaphore = asyncio.Semaphore(async_config.max_concurrent_rows)
        tasks: List[asyncio.Future] = []
        for entity_id, group_df in entity_groups:
            if str(entity_id) in skip_ids:
                continue
            tasks.append(
                self._bounded_entity_task(
                    concurrency_semaphore=concurrency_semaphore,
                    entity_id=entity_id,
                    group_df=group_df,
                    turn_processor=turn_processor,
                    text_column=column_roles.text,
                    doc_id_column=column_roles.doc_id,
                    sort_column=column_roles.sort_by,
                )
            )

        use_progress_bar = flow.display.use_progress_bar
        entity_results: List[Tuple[Any, Dict[int, str], MessyTextConversationState]] = []
        for completed in tqdm_async.as_completed(
            tasks,
            total=len(tasks),
            desc="Processing entities (conversation)",
            disable=not use_progress_bar,
        ):
            entity_results.append(await completed)

        for _entity_id, index_to_summary, _state in entity_results:
            for row_index, running_summary in index_to_summary.items():
                processed_df.at[row_index, "summary_all_context"] = running_summary
            self._checkpoint.mark_completed(_entity_id)

        entity_states: List[Tuple[Any, MessyTextConversationState]] = [
            (entity_id, state) for entity_id, _index_to_summary, state in entity_results
        ]
        return processed_df, entity_states

    async def _bounded_entity_task(
        self,
        concurrency_semaphore: asyncio.Semaphore,
        entity_id: Any,
        group_df: pd.DataFrame,
        turn_processor: AsyncMessyTextConversationTurnProcessor,
        text_column: str,
        doc_id_column: str,
        sort_column: str,
    ) -> Tuple[Any, Dict[int, str], MessyTextConversationState]:
        """Run :meth:`_process_entity` under the outer concurrency cap.

        Args:
            concurrency_semaphore (asyncio.Semaphore): Caps the number of
                entities processed in parallel.
            entity_id (Any): The entity identifier from the groupby.
            group_df (pd.DataFrame): All rows that belong to this entity.
            turn_processor (AsyncMessyTextConversationTurnProcessor):
                Per-turn processor shared across entities.
            text_column (str): Name of the text column in ``group_df``.
            doc_id_column (str): Name of the document-id column.
            sort_column (str): Name of the ordering column.

        Returns:
            Tuple[Any, Dict[int, str], MessyTextConversationState]: The
            entity id, the row-index-to-running-summary mapping, and the
            final conversation state.
        """
        async with concurrency_semaphore:
            return await self._process_entity(
                entity_id=entity_id,
                group_df=group_df,
                turn_processor=turn_processor,
                text_column=text_column,
                doc_id_column=doc_id_column,
                sort_column=sort_column,
            )

    async def _process_entity(
        self,
        entity_id: Any,
        group_df: pd.DataFrame,
        turn_processor: AsyncMessyTextConversationTurnProcessor,
        text_column: str,
        doc_id_column: str,
        sort_column: str,
    ) -> Tuple[Any, Dict[int, str], MessyTextConversationState]:
        """Sequentially process every document belonging to one entity.

        Each turn calls
        :meth:`src.processors.AsyncMessyTextConversationTurnProcessor.process_turn`
        with the current running summary as ``previous_summary``. Only
        turns whose ``info_found`` flag is truthy (or, as a fallback, whose
        summary text passes :func:`src.utils.is_informative_summary`)
        update the running summary, matching the behaviour of
        :mod:`scripts.run_summary_conversation`.

        Args:
            entity_id (Any): The entity identifier from the groupby.
            group_df (pd.DataFrame): Rows for this entity.
            turn_processor (AsyncMessyTextConversationTurnProcessor):
                Per-turn processor.
            text_column (str): Name of the text column.
            doc_id_column (str): Name of the document-id column.
            sort_column (str): Name of the ordering column.

        Returns:
            Tuple[Any, Dict[int, str], MessyTextConversationState]: The
            entity id, a mapping from the DataFrame row index to the
            running summary as of that turn, and the final conversation
            state containing every per-turn
            :class:`src.processors.ProcessorResult`.
        """
        group_sorted = group_df.sort_values(by=sort_column)
        dataframe_indices: List[int] = group_sorted.index.tolist()
        document_texts: List[str] = [str(text) for text in group_sorted[text_column]]
        document_ids: List[Any] = list(group_sorted[doc_id_column])

        running_summary: str = ""
        conversation_state = MessyTextConversationState(turn_index=0)
        per_row_running_summaries: List[str] = []

        for document_id, raw_text in zip(document_ids, document_texts):
            candidate_summary, turn_state = await turn_processor.process_turn(
                raw_text=raw_text,
                state=conversation_state,
                doc_id=document_id,
            )

            turn_result = getattr(turn_state, "last_result", None)

            has_info = False
            if turn_result is not None and hasattr(turn_result, "has_field") and turn_result.has_field("info_found"):
                info_flag = str(turn_result.get("info_found") or "").strip().lower()
                has_info = info_flag not in {"", "false", "0", "no"}
            elif turn_result is not None and hasattr(turn_result, "is_no_info"):
                has_info = not turn_result.is_no_info()
            else:
                has_info = is_informative_summary(candidate_summary)

            if has_info:
                structured_summary = (
                    turn_result.get("summary") if turn_result is not None else candidate_summary
                )
                running_summary = (structured_summary or "").strip()

            per_row_running_summaries.append(running_summary)
            conversation_state = turn_state

        index_to_running_summary: Dict[int, str] = {
            row_index: summary
            for row_index, summary in zip(dataframe_indices, per_row_running_summaries)
        }
        return entity_id, index_to_running_summary, conversation_state

    # ------------------------------------------------------------------
    # single_summary dispatcher  (matches run_summary.py)
    # ------------------------------------------------------------------

    async def _run_single_summary(
        self,
        df: pd.DataFrame,
        step: StepConfig,
    ) -> pd.DataFrame:
        """Summarise every row independently (no entity grouping).

        Mirrors the async path in ``scripts/run_summary.py``: each row
        passes through ``AsyncMessyTextProcessor.summarize_text`` with
        concurrency capped by ``max_concurrent_rows``.

        Args:
            df (pd.DataFrame): Input DataFrame.
            step (StepConfig): Step config for resource selection.

        Returns:
            pd.DataFrame: DataFrame with ``summary_all_context`` populated.
        """
        flow = self.schema.flow
        column_roles = flow.data.column_roles
        async_config = flow.async_config

        resource_id = self._select_resource_id_for_step(step)
        client = self.clients_by_id[resource_id]
        processor_config = self.processor_configs[resource_id]

        llm_semaphore: Optional[asyncio.Semaphore] = None
        if async_config.max_concurrent_llm_calls > 0:
            llm_semaphore = asyncio.Semaphore(async_config.max_concurrent_llm_calls)

        processor = AsyncMessyTextProcessor(
            client=client, config=processor_config,
            taxonomy=self.taxonomy, logger=self.logger,
            llm_semaphore=llm_semaphore,
        )

        processed_df = df.copy()
        if "summary_all_context" not in processed_df.columns:
            processed_df["summary_all_context"] = ""

        semaphore = asyncio.Semaphore(async_config.max_concurrent_rows)

        async def _summarise_row(row_index: int, text: str, doc_id: Any) -> Tuple[int, str]:
            async with semaphore:
                cleaned = processor.clean_text(text)
                if cleaned.strip():
                    summary = await processor.summarize_text(cleaned, doc_id=doc_id)
                else:
                    summary = "No relevant information found"
                return row_index, summary

        tasks = []
        for row in processed_df.itertuples():
            text = str(getattr(row, column_roles.text, ""))
            doc_id = getattr(row, column_roles.doc_id, row.Index)
            tasks.append(_summarise_row(row.Index, text, doc_id))

        use_pb = flow.display.use_progress_bar
        for completed in tqdm_async.as_completed(
            tasks, total=len(tasks),
            desc="Summarising rows", disable=not use_pb,
        ):
            row_index, summary = await completed
            processed_df.at[row_index, "summary_all_context"] = summary

        return processed_df

    # ------------------------------------------------------------------
    # classification dispatcher  (matches run_classification.py)
    # ------------------------------------------------------------------

    async def _run_classification(
        self,
        df: pd.DataFrame,
        step: StepConfig,
    ) -> pd.DataFrame:
        """Classify every row using its existing ``summary_all_context``.

        Mirrors the async path in ``scripts/run_classification.py``: each
        row's summary is classified per taxonomy key concurrently.

        Args:
            df (pd.DataFrame): Input DataFrame (must have
                ``summary_all_context``).
            step (StepConfig): Step config; ``step.keys`` can restrict
                the taxonomy keys to a subset.

        Returns:
            pd.DataFrame: DataFrame with ``{key}_classification`` columns.
        """
        flow = self.schema.flow
        async_config = flow.async_config

        resource_id = self._select_resource_id_for_step(step)
        client = self.clients_by_id[resource_id]
        processor_config = self.processor_configs[resource_id]

        llm_semaphore: Optional[asyncio.Semaphore] = None
        if async_config.max_concurrent_llm_calls > 0:
            llm_semaphore = asyncio.Semaphore(async_config.max_concurrent_llm_calls)

        processor = AsyncMessyTextProcessor(
            client=client, config=processor_config,
            taxonomy=self.taxonomy, logger=self.logger,
            llm_semaphore=llm_semaphore,
        )

        all_keys = list(self.taxonomy.get("context_definitions", {}).keys())
        if step.keys is not None and step.keys != "all":
            keys_to_classify = [k for k in step.keys if k in all_keys]
        else:
            keys_to_classify = all_keys

        processed_df = df.copy()
        for key in all_keys:
            col = f"{key}_classification"
            if col not in processed_df.columns:
                processed_df[col] = ""

        semaphore = asyncio.Semaphore(async_config.max_concurrent_rows)

        async def _classify_row(row_index: int, summary: str, doc_id: Any) -> Tuple[int, Dict[str, str]]:
            async with semaphore:
                results: Dict[str, str] = {}
                if summary and summary != "No relevant information found":
                    cls_tasks = [processor.classify_summary(summary, key) for key in keys_to_classify]
                    cls_results = await asyncio.gather(*cls_tasks)
                    for key, cls_value in zip(keys_to_classify, cls_results):
                        results[f"{key}_classification"] = cls_value
                return row_index, results

        column_roles = flow.data.column_roles
        tasks = []
        for row in processed_df.itertuples():
            summary = getattr(row, "summary_all_context", "")
            doc_id = getattr(row, column_roles.doc_id, row.Index)
            tasks.append(_classify_row(row.Index, summary, doc_id))

        use_pb = flow.display.use_progress_bar
        for completed in tqdm_async.as_completed(
            tasks, total=len(tasks),
            desc="Classifying rows", disable=not use_pb,
        ):
            row_index, cls_results = await completed
            for col, val in cls_results.items():
                processed_df.at[row_index, col] = val

        return processed_df

    # ------------------------------------------------------------------
    # label_extraction + label_summary (hybrid) dispatcher
    # ------------------------------------------------------------------

    def _build_extractors(
        self,
        step: StepConfig,
    ) -> Tuple[Dict[str, AsyncLabelExtractor], Optional[asyncio.Semaphore]]:
        """Build one ``AsyncLabelExtractor`` per taxonomy label.

        Args:
            step (StepConfig): Step config for resource selection.

        Returns:
            Tuple[Dict[str, AsyncLabelExtractor], Optional[asyncio.Semaphore]]:
            Extractor dict keyed by taxonomy label, plus the shared LLM
            semaphore (or ``None``).
        """
        flow = self.schema.flow
        async_config = flow.async_config

        resource_id = self._select_resource_id_for_step(step)
        client = self.clients_by_id[resource_id]
        processor_config = self.processor_configs[resource_id]

        llm_semaphore: Optional[asyncio.Semaphore] = None
        if async_config.max_concurrent_llm_calls > 0:
            llm_semaphore = asyncio.Semaphore(async_config.max_concurrent_llm_calls)

        context_definitions: Dict[str, str] = self.taxonomy.get("context_definitions", {})
        extractors: Dict[str, AsyncLabelExtractor] = {}
        for label_key, label_definition in context_definitions.items():
            extractors[label_key] = AsyncLabelExtractor(
                client=client, config=processor_config,
                label_key=label_key, label_definition=label_definition,
                logger=self.logger, llm_semaphore=llm_semaphore,
            )
        return extractors, llm_semaphore

    async def _run_label_hybrid(
        self,
        df: pd.DataFrame,
        extraction_step: StepConfig,
        summary_step: Optional[StepConfig],
    ) -> Tuple[pd.DataFrame, List[Tuple[Any, MessyTextConversationState]]]:
        """Hybrid label pipeline: sequential docs, concurrent labels.

        Mirrors ``_process_dataframe_hybrid`` in
        ``scripts/run_summary_conversation_by_label.py``.

        Args:
            df (pd.DataFrame): Input DataFrame.
            extraction_step (StepConfig): Config for label_extraction.
            summary_step (Optional[StepConfig]): Config for label_summary.

        Returns:
            Tuple of processed DataFrame and entity states.
        """
        flow = self.schema.flow
        column_roles = flow.data.column_roles
        async_config = flow.async_config

        extractors, llm_semaphore = self._build_extractors(extraction_step)

        summary_resource_id = self._select_resource_id_for_step(
            summary_step or extraction_step
        )
        summary_client = self.clients_by_id[summary_resource_id]
        summary_config = self.processor_configs[summary_resource_id]

        summary_processor = AsyncTextLabelsSummaryProcessor(
            client=summary_client, config=summary_config,
            taxonomy=self.taxonomy, logger=self.logger,
            llm_semaphore=llm_semaphore,
        )

        processed_df = df.copy()
        if "summary_all_context" not in processed_df.columns:
            processed_df["summary_all_context"] = ""

        entity_groups = list(processed_df.groupby(column_roles.entity_id))

        skip_ids: Set[str] = set()
        if self.resume:
            skip_ids = self._checkpoint.completed_entity_ids()
            if skip_ids:
                self.logger.info(
                    "Resuming (label hybrid): skipping %d entities.", len(skip_ids),
                )
            entity_groups = [
                (eid, gdf) for eid, gdf in entity_groups if str(eid) not in skip_ids
            ]

        concurrency_sem = asyncio.Semaphore(async_config.max_concurrent_rows)

        async def _process_entity_hybrid(
            entity_id: Any, group_df: pd.DataFrame,
        ) -> Tuple[Any, Dict[int, str], MessyTextConversationState]:
            async with concurrency_sem:
                group_sorted = group_df.sort_values(by=column_roles.sort_by)
                index_list = group_sorted.index.tolist()
                texts = [str(t) for t in group_sorted[column_roles.text]]
                doc_ids = list(group_sorted[column_roles.doc_id])

                state = MessyTextConversationState(turn_index=0)
                per_row_summaries: List[str] = []

                for doc_id, raw_text in zip(doc_ids, texts):
                    label_keys = list(extractors.keys())
                    extract_tasks = [
                        extractors[k].extract_label(text=raw_text, doc_id=doc_id)
                        for k in label_keys
                    ]
                    extract_results = await asyncio.gather(*extract_tasks)
                    label_results: Dict[str, ProcessorResult] = dict(
                        zip(label_keys, extract_results)
                    )

                    result = await summary_processor.summarize_from_labels(
                        text=raw_text, label_results=label_results,
                        previous_summary=state.last_summary, doc_id=doc_id,
                    )

                    new_results = state.results.copy()
                    new_results.append(result)
                    state = MessyTextConversationState(
                        turn_index=state.turn_index + 1, results=new_results,
                    )
                    per_row_summaries.append(result.get("summary") or "")

                idx_to_summary = dict(zip(index_list, per_row_summaries))
                return entity_id, idx_to_summary, state

        tasks = [_process_entity_hybrid(eid, gdf) for eid, gdf in entity_groups]
        use_pb = flow.display.use_progress_bar
        entity_results: List[Tuple[Any, Dict[int, str], MessyTextConversationState]] = []
        for completed in tqdm_async.as_completed(
            tasks, total=len(tasks),
            desc="Processing entities (label hybrid)", disable=not use_pb,
        ):
            entity_results.append(await completed)

        for _eid, idx_to_summary, _st in entity_results:
            for row_idx, summary in idx_to_summary.items():
                processed_df.at[row_idx, "summary_all_context"] = summary
            self._checkpoint.mark_completed(_eid)

        entity_states = [(eid, st) for eid, _, st in entity_results]
        return processed_df, entity_states

    # ------------------------------------------------------------------
    # label_extraction + label_summary (full_async) dispatcher
    # ------------------------------------------------------------------

    async def _run_label_full_async(
        self,
        df: pd.DataFrame,
        extraction_step: StepConfig,
        summary_step: Optional[StepConfig],
    ) -> Tuple[pd.DataFrame, List[Tuple[Any, MessyTextConversationState]]]:
        """Full-async label pipeline: all docs concurrent, then synthesis.

        Mirrors ``_process_dataframe_full_async`` in
        ``scripts/run_summary_conversation_by_label.py``.

        Args:
            df (pd.DataFrame): Input DataFrame.
            extraction_step (StepConfig): Config for label_extraction.
            summary_step (Optional[StepConfig]): Config for label_summary.

        Returns:
            Tuple of processed DataFrame and entity states.
        """
        flow = self.schema.flow
        column_roles = flow.data.column_roles
        async_config = flow.async_config

        extractors, llm_semaphore = self._build_extractors(extraction_step)

        summary_resource_id = self._select_resource_id_for_step(
            summary_step or extraction_step
        )
        summary_client = self.clients_by_id[summary_resource_id]
        summary_config = self.processor_configs[summary_resource_id]

        summary_processor = AsyncTextLabelsSummaryProcessor(
            client=summary_client, config=summary_config,
            taxonomy=self.taxonomy, logger=self.logger,
            llm_semaphore=llm_semaphore,
        )
        orchestrator = AsyncTextConversationOrchestrator(summary_processor)

        processed_df = df.copy()
        if "summary_all_context" not in processed_df.columns:
            processed_df["summary_all_context"] = ""

        entity_groups = list(processed_df.groupby(column_roles.entity_id))

        skip_ids: Set[str] = set()
        if self.resume:
            skip_ids = self._checkpoint.completed_entity_ids()
            if skip_ids:
                self.logger.info(
                    "Resuming (label full-async): skipping %d entities.", len(skip_ids),
                )
            entity_groups = [
                (eid, gdf) for eid, gdf in entity_groups if str(eid) not in skip_ids
            ]

        concurrency_sem = asyncio.Semaphore(async_config.max_concurrent_rows)

        async def _process_entity_full_async(
            entity_id: Any, group_df: pd.DataFrame,
        ) -> Tuple[Any, Dict[int, str], MessyTextConversationState]:
            async with concurrency_sem:
                group_sorted = group_df.sort_values(by=column_roles.sort_by)
                index_list = group_sorted.index.tolist()
                texts = [str(t) for t in group_sorted[column_roles.text]]
                doc_ids = list(group_sorted[column_roles.doc_id])

                label_keys = list(extractors.keys())
                n_labels = len(label_keys)
                all_extract_tasks = []
                for doc_id, text in zip(doc_ids, texts):
                    for k in label_keys:
                        all_extract_tasks.append(
                            extractors[k].extract_label(text=text, doc_id=doc_id)
                        )
                all_extract_results = await asyncio.gather(*all_extract_tasks)

                per_doc_label_results: List[Dict[str, ProcessorResult]] = []
                for i in range(len(texts)):
                    offset = i * n_labels
                    per_doc_label_results.append({
                        label_keys[j]: all_extract_results[offset + j]
                        for j in range(n_labels)
                    })

                documents: List[Tuple[str, Dict[str, ProcessorResult], Any]] = [
                    (text, lr, did)
                    for text, lr, did in zip(texts, per_doc_label_results, doc_ids)
                ]

                _summaries, state = await orchestrator.run_conversation(
                    documents=documents,
                    use_progress_bar=False,
                )

                per_doc_summaries = [
                    str(r.get("summary") or "") for r in state.results
                ]
                synthesis_result = await summary_processor.synthesize_from_summaries(
                    per_doc_summaries=per_doc_summaries, doc_id=entity_id,
                )

                new_results = state.results.copy()
                new_results.append(synthesis_result)
                state = MessyTextConversationState(
                    turn_index=state.turn_index + 1, results=new_results,
                )

                final_summary = synthesis_result.get("summary") or ""
                idx_to_summary = {row_idx: final_summary for row_idx in index_list}
                return entity_id, idx_to_summary, state

        tasks = [_process_entity_full_async(eid, gdf) for eid, gdf in entity_groups]
        use_pb = flow.display.use_progress_bar
        entity_results: List[Tuple[Any, Dict[int, str], MessyTextConversationState]] = []
        for completed in tqdm_async.as_completed(
            tasks, total=len(tasks),
            desc="Processing entities (label full-async)", disable=not use_pb,
        ):
            entity_results.append(await completed)

        for _eid, idx_to_summary, _st in entity_results:
            for row_idx, summary in idx_to_summary.items():
                processed_df.at[row_idx, "summary_all_context"] = summary
            self._checkpoint.mark_completed(_eid)

        entity_states = [(eid, st) for eid, _, st in entity_results]
        return processed_df, entity_states

    # ------------------------------------------------------------------
    # evaluation dispatcher  (matches run_evaluation.py)
    # ------------------------------------------------------------------

    async def _run_evaluation(
        self,
        df: pd.DataFrame,
        step: StepConfig,
    ) -> pd.DataFrame:
        """Run evaluation benchmarks on existing summaries.

        The evaluation step delegates to the existing evaluator classes in
        ``src.processors`` / ``scripts.run_evaluation``. Because
        evaluators have significant external dependencies (SummaC, NLTK,
        sklearn) that may not always be installed, this dispatcher
        imports lazily and logs a warning if a benchmark cannot be loaded.

        Currently supported benchmarks via ``step.benchmarks``:

        - ``geval_summarization``: LLM-based G-Eval score.
        - ``geval_hallucination``: LLM-based hallucination score.

        SummaC and DefaultMetrics require heavy local dependencies
        (``summac``, ``nltk``). These are imported lazily; if unavailable
        the benchmark is skipped with a logged warning.

        Args:
            df (pd.DataFrame): DataFrame with ``summary_all_context``.
            step (StepConfig): Step config with ``benchmarks`` dict.

        Returns:
            pd.DataFrame: DataFrame with evaluation score columns added.
        """
        if not step.benchmarks:
            self.logger.info("Evaluation step has no benchmarks configured; skipping.")
            return df

        resource_id = self._select_resource_id_for_step(step)
        client = self.clients_by_id[resource_id]
        processor_config = self.processor_configs[resource_id]
        flow = self.schema.flow
        async_config = flow.async_config
        column_roles = flow.data.column_roles

        processed_df = df.copy()
        valid_mask = (
            processed_df["summary_all_context"].notna()
            & (processed_df["summary_all_context"].str.strip() != "")
            & processed_df[column_roles.text].notna()
            & (processed_df[column_roles.text].astype(str).str.strip() != "")
        )
        df_eval = processed_df[valid_mask].copy()
        if df_eval.empty:
            self.logger.warning("No rows with valid text + summary for evaluation.")
            return processed_df

        eval_semaphore = asyncio.Semaphore(async_config.max_concurrent_rows)

        for benchmark_name, enabled in step.benchmarks.items():
            if not enabled:
                continue

            if benchmark_name in {"geval_summarization", "geval_hallucination"}:
                try:
                    from src.processors import AsyncGEvalEvaluator  # type: ignore[attr-defined]
                except ImportError:
                    self.logger.warning(
                        "AsyncGEvalEvaluator not available; skipping %s.",
                        benchmark_name,
                    )
                    continue

                evaluator = AsyncGEvalEvaluator(client, processor_config, self.logger)
                method_name = (
                    "evaluate_summarization"
                    if benchmark_name == "geval_summarization"
                    else "evaluate_hallucination"
                )
                eval_method = getattr(evaluator, method_name)
                col_name = f"{benchmark_name}_score"

                async def _eval_row(idx: int, source: str, summary: str) -> Tuple[int, float]:
                    async with eval_semaphore:
                        score = await eval_method(source, summary)
                        return idx, float(score)

                eval_tasks = [
                    _eval_row(
                        idx,
                        str(row[column_roles.text]),
                        str(row["summary_all_context"]),
                    )
                    for idx, row in df_eval.iterrows()
                ]
                results_map: Dict[int, float] = {}
                for completed in tqdm_async.as_completed(
                    eval_tasks, total=len(eval_tasks),
                    desc=benchmark_name, disable=not flow.display.use_progress_bar,
                ):
                    idx, score = await completed
                    results_map[idx] = score

                processed_df[col_name] = processed_df.index.map(
                    lambda i: results_map.get(i)  # noqa: B023
                )
                self.logger.info(
                    "Benchmark %s complete: %d rows scored.",
                    benchmark_name, len(results_map),
                )
            else:
                self.logger.warning(
                    "Benchmark %r is not implemented in the flow runner. "
                    "Use scripts/run_evaluation.py for SummaC or DefaultMetrics.",
                    benchmark_name,
                )

        return processed_df

    def _write_outputs(
        self,
        processed_df: pd.DataFrame,
        entity_states: List[Tuple[Any, MessyTextConversationState]],
        model_name: str,
    ) -> None:
        """Write every output CSV declared in :class:`src.flow_loader.OutputConfig`.

        Creates parent directories as needed. The per-row summary CSV is
        always written. The per-turn results, per-entity states, and
        flattened spans CSVs are written only when their output paths are
        configured.

        Args:
            processed_df (pd.DataFrame): DataFrame augmented with
                ``summary_all_context`` and ``model``.
            entity_states (List[Tuple[Any, MessyTextConversationState]]):
                Per-entity conversation states.
            model_name (str): Model identifier used for the ``model``
                column and the extend-mode replacement key.

        Returns:
            None.
        """
        output_config = self.schema.flow.output
        processed_df = processed_df.copy()
        processed_df["model"] = model_name
        if "summary_all_context" in processed_df.columns:
            processed_df["summary_all_context"] = processed_df["summary_all_context"].replace(
                ["No information", "No relevant information found"],
                "",
            )

        summary_path = Path(output_config.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        if output_config.extend and summary_path.exists():
            existing_df = pd.read_csv(summary_path, encoding="utf-8")
            if "model" in existing_df.columns:
                existing_df = existing_df[existing_df["model"] != model_name]
            combined_df = pd.concat([existing_df, processed_df], ignore_index=True)
            combined_df.to_csv(summary_path, index=False, encoding="utf-8")
            self.logger.info(
                "Summary output extended: %d existing + %d new = %d total rows",
                len(existing_df),
                len(processed_df),
                len(combined_df),
            )
        else:
            processed_df.to_csv(summary_path, index=False, encoding="utf-8")
            self.logger.info(
                "Summary output saved to %s (%d rows)",
                summary_path,
                len(processed_df),
            )

        if not entity_states:
            return

        results_rows: List[Dict[str, Any]] = []
        states_rows: List[Dict[str, Any]] = []
        spans_rows: List[Dict[str, Any]] = []

        for entity_id, state in entity_states:
            states_rows.append(
                serialize_state_entry(
                    state=state,
                    victim_id=str(entity_id),
                    model_name=model_name,
                )
            )
            spans_rows.extend(
                flatten_spans_from_state(
                    state=state,
                    victim_id=str(entity_id),
                    model_name=model_name,
                )
            )
            for turn_index, turn_result in enumerate(state.results):
                results_rows.append(
                    serialize_result_entry(
                        result=turn_result,
                        victim_id=str(entity_id),
                        model_name=model_name,
                        turn_index=turn_index,
                    )
                )

        if output_config.results_csv is not None:
            results_path = Path(output_config.results_csv)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            write_results(
                rows=results_rows,
                path=results_path,
                extend=output_config.extend,
                model_name=model_name,
            )
            self.logger.info(
                "Results records saved to %s (%d rows)",
                results_path,
                len(results_rows),
            )

        if output_config.states_csv is not None:
            states_path = Path(output_config.states_csv)
            states_path.parent.mkdir(parents=True, exist_ok=True)
            write_states(
                rows=states_rows,
                path=states_path,
                extend=output_config.extend,
                model_name=model_name,
            )
            self.logger.info(
                "State records saved to %s (%d rows)",
                states_path,
                len(states_rows),
            )

        if output_config.spans_csv is not None:
            spans_path = Path(output_config.spans_csv)
            spans_path.parent.mkdir(parents=True, exist_ok=True)
            write_spans(
                rows=spans_rows,
                path=spans_path,
                extend=output_config.extend,
                model_name=model_name,
            )
            self.logger.info(
                "Span records saved to %s (%d rows)",
                spans_path,
                len(spans_rows),
            )


def build_flow(flow_yaml_path: Path, resume: bool = False) -> FlowRunner:
    """Build a :class:`FlowRunner` from a flow YAML file.

    Args:
        flow_yaml_path (Path): Path to the flow YAML file on disk.
        resume (bool): When ``True`` the runner loads entity-level
            checkpoints and skips already-completed entities.

    Returns:
        FlowRunner: The configured runner.

    Raises:
        FileNotFoundError: If the flow YAML, taxonomy, or prompts file is
            missing.
        ValueError: If schema validation fails or an ``api_key_env``
            variable is not present in the environment.
    """
    _load_dotenv_into_environ(_PROJECT_ROOT)

    schema = FlowSchema.load_from_path(Path(flow_yaml_path))
    flow_config = schema.flow

    taxonomy_payload = _load_json_file(Path(flow_config.taxonomy))
    prompts_payload = _load_json_file(Path(flow_config.prompts))

    logger = setup_logger(log_file=flow_config.logging.file)

    resolved_resources: Dict[str, LLMResource] = {}
    clients_by_id: Dict[str, AsyncOpenAI] = {}
    processor_configs: Dict[str, Dict[str, Any]] = {}

    for raw_resource in flow_config.resources:
        resolved_resource = _resolve_resource_credentials(raw_resource)
        resolved_resources[resolved_resource.id] = resolved_resource
        clients_by_id[resolved_resource.id] = _build_async_client(
            resource=resolved_resource,
            max_retries=flow_config.async_config.max_retries,
        )
        processor_configs[resolved_resource.id] = _build_processor_runtime_config(
            resource=resolved_resource,
            prompts_payload=prompts_payload,
            logging_config=flow_config.logging,
        )

    logger.info(
        "Flow %r loaded with %d resource(s) and %d step(s).",
        flow_config.name,
        len(resolved_resources),
        len(flow_config.steps),
    )

    return FlowRunner(
        schema=schema,
        resolved_resources=resolved_resources,
        clients_by_id=clients_by_id,
        processor_configs=processor_configs,
        taxonomy=taxonomy_payload,
        prompts=prompts_payload,
        logger=logger,
        resume=resume,
    )
