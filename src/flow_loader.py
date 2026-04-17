"""Loader for MessyText configuration-driven flow YAML files.

The *schema* itself lives in the YAML documents under ``config/flows/``. This
module is the actor that reads one of those YAML files from disk, applies the
``llm:`` shorthand normalisation, and validates the result against a set of
Pydantic models. It produces a typed :class:`FlowSchema` Python object that
:mod:`src.flow_builder` consumes.

The Pydantic models declared here are the *validation rules* for the YAML
schema — not the schema. Validation runs before any LLM call is made, so
typos and missing fields fail fast with clear error messages instead of
surfacing as runtime errors deep inside the processors.

Contents and relationships
--------------------------

- :class:`FlowSchema` — top-level envelope matching the YAML file. Wraps a
  single :class:`FlowConfig` under the ``flow:`` key. Exposes
  :meth:`FlowSchema.load_from_path` which reads the YAML from disk, applies
  the ``llm:`` shorthand normalisation, and returns the validated object.
- :class:`FlowConfig` — the ``flow:`` block. Holds schema version, metadata,
  LLM resources, data source, taxonomy and prompt paths, ordered pipeline
  steps, async settings, output paths, logging, and display options.
- :class:`LLMResource` — one named LLM provider entry under
  ``flow.resources[*]``. Each resource is addressable by ``id`` and is
  referenced by :attr:`StepConfig.llm` when a step needs a non-default LLM.
- :class:`LLMShorthand` — the single-LLM sugar block accepted as
  ``flow.llm:``. The loader rewrites this into a one-entry ``resources``
  list with ``id = "default"`` so downstream code only deals with the
  canonical :class:`LLMResource` list.
- :class:`ColumnRoles` — maps the user's actual CSV column names to the
  internal roles (``text``, ``entity_id``, ``doc_id``, ``sort_by``,
  ``passthrough``) that the runner loops use.
- :class:`DataConfig` — the ``flow.data:`` block pointing at the input CSV
  and holding the :class:`ColumnRoles` binding.
- :class:`StepConfig` — a single processing step. ``type`` selects the
  runtime behaviour and is restricted to the registered step types. Optional
  fields such as :attr:`StepConfig.mode` and
  :attr:`StepConfig.benchmarks` are only meaningful for specific step types
  and are validated accordingly.
- :class:`AsyncConfig`, :class:`OutputConfig`, :class:`LoggingConfig`, and
  :class:`DisplayConfig` — the remaining runtime sections that control
  concurrency, where results are written, what is logged, and whether
  progress bars are shown.

How the rest of the system uses this module
-------------------------------------------

:func:`src.flow_builder.build_flow` calls
:meth:`FlowSchema.load_from_path` to obtain a validated schema, then walks
:attr:`FlowConfig.resources` to construct OpenAI-compatible clients, walks
:attr:`FlowConfig.steps` to instantiate the matching processor classes from
:mod:`src.processors`, and uses :attr:`FlowConfig.data`,
:attr:`FlowConfig.async_config`, and :attr:`FlowConfig.output` to drive the
runner loops.

Invariants enforced by this module
----------------------------------

- At least one :class:`LLMResource` exists after normalisation, and every
  resource has a unique ``id``.
- Every :attr:`StepConfig.llm` string resolves to an existing resource
  ``id``. Steps that omit ``llm`` are wired to the resource with ``id =
  "default"`` by the builder.
- :attr:`StepConfig.type` is one of the values registered in
  :data:`STEP_TYPES`.
- :attr:`StepConfig.unit` is one of ``row`` / ``document`` / ``entity``.
- :class:`ColumnRoles` requires ``text``, ``entity_id``, ``doc_id``, and
  ``sort_by`` to be present.
- :attr:`LLMResource.api_key_env` and :attr:`LLMResource.api_key` are
  mutually exclusive in the YAML: the builder resolves ``api_key_env``
  against ``os.environ`` at startup and the runtime copy of the resource
  ends up with ``api_key`` populated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


STEP_TYPES: frozenset[str] = frozenset(
    {
        "single_summary",
        "conversation_summary_first",
        "conversation_summary_update",
        "label_extraction",
        "label_summary",
        "classification",
        "evaluation",
    }
)
"""Registered step ``type`` values accepted by :class:`StepConfig`."""


UNIT_VALUES: frozenset[str] = frozenset({"row", "document", "entity"})
"""Registered step ``unit`` values accepted by :class:`StepConfig`."""


PROVIDER_VALUES: frozenset[str] = frozenset({"local_vllm", "openrouter", "openai"})
"""Registered provider values accepted by :class:`LLMResource`."""


PROVIDER_DEFAULT_API_BASE: Dict[str, str] = {
    "local_vllm": "http://localhost:8000/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}
"""Default ``api_base`` per provider. Used by the builder when the YAML
omits ``api_base`` on a resource."""


class LLMResource(BaseModel):
    """One named LLM provider entry addressable by :attr:`id`.

    A resource describes a single OpenAI-compatible endpoint plus the
    generation parameters that :class:`src.processors.AsyncMessyTextProcessor`
    reads from its runtime config dict. The builder constructs one
    ``AsyncOpenAI`` client per resource and wires each step to the client
    whose resource ``id`` the step references.

    Attributes:
        id (str): Unique identifier referenced by
            :attr:`StepConfig.llm`. The shorthand block is normalised into a
            resource with ``id = "default"``.
        type (Literal["llm_provider"]): Discriminator kept for future
            resource kinds. Always ``"llm_provider"`` in Phase 1.
        provider (str): One of :data:`PROVIDER_VALUES`. Selects the default
            ``api_base`` via :data:`PROVIDER_DEFAULT_API_BASE` when
            ``api_base`` is omitted.
        model (str): Model identifier sent to the endpoint as
            ``chat.completions.create(model=...)``.
        api_base (Optional[str]): OpenAI-compatible base URL. Defaults to
            the provider-specific base URL when omitted.
        api_key (Optional[str]): Literal API key. Only used for
            ``local_vllm`` where the key is ``"dummy"``. For hosted
            providers, use :attr:`api_key_env` instead.
        api_key_env (Optional[str]): Name of the environment variable that
            holds the real API key. Resolved by the builder from the
            project-root ``.env`` file. Mutually exclusive with
            :attr:`api_key`.
        temperature (float): Temperature forwarded to every LLM call
            produced against this resource.
        max_tokens_summary (int): ``max_tokens`` for summary-style calls.
        max_tokens_classification (int): ``max_tokens`` for classification
            calls.

    Methods:
        validate_provider: Ensure :attr:`provider` is a registered value.
        validate_api_key_shape: Ensure ``api_key`` and ``api_key_env`` are
            not both set at the same time.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["llm_provider"] = "llm_provider"
    provider: str
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0
    max_tokens_summary: int = 1024
    max_tokens_classification: int = 256

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, provider_value: str) -> str:
        """Ensure the provider value is one registered in :data:`PROVIDER_VALUES`.

        Args:
            provider_value (str): The raw provider string as read from YAML.

        Returns:
            str: The provider value, unchanged.

        Raises:
            ValueError: If ``provider_value`` is not in
                :data:`PROVIDER_VALUES`.
        """
        if provider_value not in PROVIDER_VALUES:
            raise ValueError(
                f"provider must be one of {sorted(PROVIDER_VALUES)}; "
                f"got {provider_value!r}"
            )
        return provider_value

    @model_validator(mode="after")
    def validate_api_key_shape(self) -> "LLMResource":
        """Disallow declaring both :attr:`api_key` and :attr:`api_key_env`.

        Returns:
            LLMResource: The same instance, unchanged.

        Raises:
            ValueError: If both ``api_key`` and ``api_key_env`` are set.
        """
        if self.api_key is not None and self.api_key_env is not None:
            raise ValueError(
                f"Resource {self.id!r} declares both api_key and api_key_env; "
                "use exactly one. For hosted providers prefer api_key_env so "
                "the secret stays in .env."
            )
        return self


class LLMShorthand(BaseModel):
    """Single-LLM sugar block accepted as ``flow.llm:``.

    This object exists only to support the shorthand form of the schema. The
    loader normalises it into a one-entry :class:`LLMResource` list with
    ``id = "default"`` before any downstream code runs.

    Attributes:
        provider (str): Same meaning as :attr:`LLMResource.provider`.
        model (str): Same meaning as :attr:`LLMResource.model`.
        api_base (Optional[str]): Same meaning as :attr:`LLMResource.api_base`.
        api_key (Optional[str]): Same meaning as :attr:`LLMResource.api_key`.
        api_key_env (Optional[str]): Same meaning as
            :attr:`LLMResource.api_key_env`.
        temperature (float): Same meaning as :attr:`LLMResource.temperature`.
        max_tokens_summary (int): Same meaning as
            :attr:`LLMResource.max_tokens_summary`.
        max_tokens_classification (int): Same meaning as
            :attr:`LLMResource.max_tokens_classification`.

    Methods:
        to_resource: Convert this shorthand block into a canonical
            :class:`LLMResource` with ``id = "default"``.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0
    max_tokens_summary: int = 1024
    max_tokens_classification: int = 256

    def to_resource(self) -> LLMResource:
        """Promote this shorthand block into a canonical :class:`LLMResource`.

        Returns:
            LLMResource: A resource with ``id = "default"`` and
            ``type = "llm_provider"``, carrying every value from this
            shorthand block.
        """
        return LLMResource(
            id="default",
            type="llm_provider",
            provider=self.provider,
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            api_key_env=self.api_key_env,
            temperature=self.temperature,
            max_tokens_summary=self.max_tokens_summary,
            max_tokens_classification=self.max_tokens_classification,
        )


class ColumnRoles(BaseModel):
    """Mapping from user CSV column names to the pipeline's internal roles.

    The runner never hard-codes column names. Instead, every loop reads
    column names out of this object. A user whose CSV has columns named
    ``article_body``, ``case_number``, ``report_id``, ``pub_date`` sets the
    four role fields accordingly, and the processors see the same
    ``text: str`` / ``doc_id: Any`` arguments they always see.

    Attributes:
        text (str): Name of the column that contains the raw document text
            fed to the LLM.
        entity_id (str): Name of the column that groups rows belonging to
            the same entity. For the current dataset this is ``victim``.
        doc_id (str): Name of the column that uniquely identifies each
            document. Passed through to :class:`src.processors.ProcessorResult`
            as ``doc_id`` for traceability.
        sort_by (str): Name of the column used to order documents within
            an entity group before sequential processing.
        passthrough (List[str]): Extra column names the user wants kept
            untouched in the output CSVs.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    entity_id: str
    doc_id: str
    sort_by: str
    passthrough: List[str] = Field(default_factory=list)


class DataConfig(BaseModel):
    """``flow.data:`` block: input CSV path and column role binding.

    Attributes:
        input_csv (str): Path to the CSV file the runner loads at startup.
            Relative paths are resolved against the current working
            directory, matching existing scripts.
        column_roles (ColumnRoles): The column-to-role mapping used by
            every runner loop.
    """

    model_config = ConfigDict(extra="forbid")

    input_csv: str
    column_roles: ColumnRoles


class StepConfig(BaseModel):
    """One processing step in :attr:`FlowConfig.steps`.

    ``type`` selects the runtime behaviour. Optional fields are only
    meaningful for specific step types:

    - ``mode`` applies to ``label_summary`` (``hybrid`` or ``full_async``).
    - ``keys`` applies to ``classification`` (``"all"`` or a list of
      taxonomy keys).
    - ``benchmarks`` applies to ``evaluation`` (dict of benchmark flags).

    Attributes:
        type (str): One of :data:`STEP_TYPES`. Selects the processor class
            and runner loop.
        unit (str): One of :data:`UNIT_VALUES`. Declares the unit of
            analysis the step operates on.
        group_by (Optional[str]): Grouping rule. Currently accepts
            ``"entity"`` for grouped loops; ``None`` means the runner
            iterates rows directly.
        llm (Optional[str]): Resource ``id`` this step uses. ``None`` means
            the builder wires the step to the resource with
            ``id = "default"``.
        mode (Optional[str]): ``hybrid`` or ``full_async`` for
            ``label_summary``.
        keys (Optional[Any]): ``"all"`` or a list of taxonomy keys for
            ``classification``.
        benchmarks (Optional[Dict[str, Any]]): Benchmark flag dict for
            ``evaluation``.

    Methods:
        validate_type: Ensure :attr:`type` is in :data:`STEP_TYPES`.
        validate_unit: Ensure :attr:`unit` is in :data:`UNIT_VALUES`.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    unit: str
    group_by: Optional[str] = None
    llm: Optional[str] = None
    mode: Optional[str] = None
    keys: Optional[Any] = None
    benchmarks: Optional[Dict[str, Any]] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, step_type_value: str) -> str:
        """Ensure the step type is registered in :data:`STEP_TYPES`.

        Args:
            step_type_value (str): Raw step type string from YAML.

        Returns:
            str: The step type, unchanged.

        Raises:
            ValueError: If the step type is not registered.
        """
        if step_type_value not in STEP_TYPES:
            raise ValueError(
                f"step type must be one of {sorted(STEP_TYPES)}; "
                f"got {step_type_value!r}"
            )
        return step_type_value

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, unit_value: str) -> str:
        """Ensure the unit is one of :data:`UNIT_VALUES`.

        Args:
            unit_value (str): Raw unit string from YAML.

        Returns:
            str: The unit, unchanged.

        Raises:
            ValueError: If the unit is not registered.
        """
        if unit_value not in UNIT_VALUES:
            raise ValueError(
                f"unit must be one of {sorted(UNIT_VALUES)}; "
                f"got {unit_value!r}"
            )
        return unit_value


class AsyncConfig(BaseModel):
    """``flow.async:`` block: concurrency and retry settings.

    Attributes:
        enabled (bool): When ``True`` the runner uses the async code path;
            otherwise the synchronous path is used.
        max_concurrent_rows (int): Upper bound on concurrent entity-level
            or row-level tasks. Matches ``max_concurrent_rows`` used by
            the existing scripts.
        max_concurrent_llm_calls (int): Upper bound on simultaneous
            in-flight LLM requests. Passed as an ``asyncio.Semaphore`` to
            every processor constructor that accepts ``llm_semaphore``.
        max_retries (int): Passed to ``AsyncOpenAI(max_retries=...)``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_concurrent_rows: int = 15
    max_concurrent_llm_calls: int = 50
    max_retries: int = 5


class OutputConfig(BaseModel):
    """``flow.output:`` block: where runner results are written.

    Attributes:
        summary_csv (str): Path for the per-row summary CSV.
        results_csv (Optional[str]): Path for the flattened per-turn
            result CSV. ``None`` disables results output.
        states_csv (Optional[str]): Path for the per-entity serialized
            state CSV. ``None`` disables states output.
        spans_csv (Optional[str]): Path for the flattened spans CSV.
            ``None`` disables spans output.
        extend (bool): When ``True`` the writer appends to existing files
            while replacing rows for the current model.
    """

    model_config = ConfigDict(extra="forbid")

    summary_csv: str
    results_csv: Optional[str] = None
    states_csv: Optional[str] = None
    spans_csv: Optional[str] = None
    extend: bool = False


class LoggingConfig(BaseModel):
    """``flow.logging:`` block: logger file path and verbosity flags.

    Attributes:
        file (str): Path to the log file.
        log_progress (bool): When ``True`` the runner logs per-step
            progress messages.
        log_prompts (bool): When ``True`` processors log every constructed
            prompt.
        log_response (bool): When ``True`` processors log every raw LLM
            response. Matches the flag read by
            :class:`src.processors.AsyncMessyTextConversationTurnProcessor`.
    """

    model_config = ConfigDict(extra="forbid")

    file: str = "processing.log"
    log_progress: bool = True
    log_prompts: bool = False
    log_response: bool = False


class DisplayConfig(BaseModel):
    """``flow.display:`` block: user-visible progress options.

    Attributes:
        use_progress_bar (bool): When ``True`` the runner shows tqdm
            progress bars during processing.
    """

    model_config = ConfigDict(extra="forbid")

    use_progress_bar: bool = True


class FlowConfig(BaseModel):
    """The ``flow:`` block: the full validated flow definition.

    Attributes:
        schema_version (int): Version of the YAML format. Incremented when
            the schema changes in an incompatible way.
        name (str): Short identifier for the flow. Used in logs and output
            file prefixes.
        description (str): Free-form human-readable description.
        resources (List[LLMResource]): Named LLM provider entries. Either
            declared directly under ``resources:`` or produced from the
            ``llm:`` shorthand.
        data (DataConfig): Input CSV and column role binding.
        taxonomy (str): Path to the taxonomy JSON file consumed by the
            processors.
        prompts (str): Path to the prompts JSON file consumed by the
            processors.
        steps (List[StepConfig]): Ordered pipeline steps.
        processing_limit (Optional[int]): When set, caps the number of
            entities (grouped pipelines) or rows (flat pipelines) the
            runner processes. ``None`` means process all.
        async_config (AsyncConfig): Concurrency settings. Aliased as
            ``async`` in the YAML.
        output (OutputConfig): Output CSV paths.
        logging (LoggingConfig): Logger file and verbosity flags.
        display (DisplayConfig): User-visible progress options.

    Methods:
        validate_resources: Ensure resource ids are unique and that every
            ``step.llm`` reference resolves.
        validate_steps_non_empty: Ensure at least one step is declared.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int
    name: str
    description: str = ""
    resources: List[LLMResource]
    data: DataConfig
    taxonomy: str
    prompts: str
    steps: List[StepConfig]
    processing_limit: Optional[int] = None
    async_config: AsyncConfig = Field(default_factory=AsyncConfig, alias="async")
    output: OutputConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)

    @model_validator(mode="after")
    def validate_resources(self) -> "FlowConfig":
        """Enforce unique resource ids and resolvable ``step.llm`` references.

        Returns:
            FlowConfig: The same instance, unchanged.

        Raises:
            ValueError: If a duplicate resource id is found or a step
                references an unknown resource id.
        """
        seen_ids: set[str] = set()
        for resource in self.resources:
            if resource.id in seen_ids:
                raise ValueError(
                    f"Duplicate LLM resource id: {resource.id!r}. "
                    "Every resource must have a unique id."
                )
            seen_ids.add(resource.id)

        if not self.resources:
            raise ValueError(
                "At least one LLM resource must be declared (either under "
                "flow.resources[*] or via the flow.llm shorthand)."
            )

        for step_index, step in enumerate(self.steps):
            if step.llm is not None and step.llm not in seen_ids:
                raise ValueError(
                    f"Step {step_index} (type={step.type!r}) references "
                    f"llm={step.llm!r}, which is not among the declared "
                    f"resource ids: {sorted(seen_ids)}"
                )
        return self

    @model_validator(mode="after")
    def validate_steps_non_empty(self) -> "FlowConfig":
        """Reject a flow with zero steps.

        Returns:
            FlowConfig: The same instance, unchanged.

        Raises:
            ValueError: If :attr:`steps` is empty.
        """
        if not self.steps:
            raise ValueError("flow.steps must contain at least one step.")
        return self


class FlowSchema(BaseModel):
    """Top-level envelope matching the YAML file structure.

    Attributes:
        flow (FlowConfig): The validated flow definition.

    Methods:
        load_from_path: Read a YAML file from disk, apply shorthand
            normalisation, and return the validated :class:`FlowSchema`.
    """

    model_config = ConfigDict(extra="forbid")

    flow: FlowConfig

    @classmethod
    def load_from_path(cls, flow_yaml_path: Path) -> "FlowSchema":
        """Load and validate a flow YAML file.

        The loader applies the ``llm:`` shorthand normalisation before
        validation: if the YAML declares ``flow.llm:`` but not
        ``flow.resources:``, the ``llm:`` block is promoted to a one-entry
        ``resources`` list with ``id = "default"``. Declaring both
        ``flow.llm`` and ``flow.resources`` is rejected to avoid ambiguity.

        Args:
            flow_yaml_path (Path): Path to the flow YAML file on disk.

        Returns:
            FlowSchema: The validated schema with ``resources`` always
            populated.

        Raises:
            FileNotFoundError: If ``flow_yaml_path`` does not exist.
            ValueError: If the YAML declares both ``flow.llm`` and
                ``flow.resources``, or if Pydantic validation fails.
        """
        flow_yaml_path = Path(flow_yaml_path)
        if not flow_yaml_path.exists():
            raise FileNotFoundError(
                f"Flow YAML file not found: {flow_yaml_path}"
            )

        with flow_yaml_path.open("r", encoding="utf-8") as yaml_file:
            raw_document: Dict[str, Any] = yaml.safe_load(yaml_file) or {}

        flow_block: Dict[str, Any] = raw_document.get("flow") or {}
        has_shorthand = "llm" in flow_block
        has_resources = "resources" in flow_block

        if has_shorthand and has_resources:
            raise ValueError(
                "flow.llm (shorthand) and flow.resources (canonical) are "
                "mutually exclusive. Use exactly one of them."
            )

        if has_shorthand:
            shorthand_block = LLMShorthand(**flow_block["llm"])
            flow_block = dict(flow_block)
            del flow_block["llm"]
            flow_block["resources"] = [shorthand_block.to_resource().model_dump()]
            raw_document = dict(raw_document)
            raw_document["flow"] = flow_block

        return cls.model_validate(raw_document)
