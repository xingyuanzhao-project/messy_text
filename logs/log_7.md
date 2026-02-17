1. Label Grouping (4 classes, 15 labels total)

| Class | Subclass | Labels |
|-------|----------|--------|
| Entity | Perpetrator | `perp_tipo1`, `perp_tipo2`, `amenaza_quien` |
| Entity | Victim | `vic_grupo_social`, `soc_civil` |
| Event | Victim Experience | `captura_metodo`, `captura_tipo`, `cautiverio_trato`, `desenlace`, `desenlace_tipo` |
| Event | Legal Procedure | `proced_contacto1`, `proced_contacto2`, `proced_contactado`, `proced_sent_tipo`, `Tribunal_tipo` |

Purpose: answer "what does this label describe?" at a glance.

2. 
after that: draft/outline of the paper

3
do i do trace back? like, what is the original spans? can be done in the fields. just add a field "original_span" or something like that next to the summary_by_label field. and, it should add the list of spans, if found more.

or thinking more about the output field, or aka the storage format. 

maybe thinking, how to "pass" things, should i do state or result object?

3.1 proposed snippet, modifications only.:

3.1.1 State object:
```python
@dataclass
class MessyTextConversationState:
    turn_index: int = 0
    results: List[ProcessorResult] = field(default_factory=list)

    @property
    def last_result(self) -> Optional[ProcessorResult]:
        return self.results[-1] if self.results else None

    @property
    def last_summary(self) -> str:
        if not self.results:
            return ""
        return self.results[-1].get("summary") or ""
```

```python
def process_turn(
    self,
    raw_text: str,
    state: Optional[MessyTextConversationState] = None,
    doc_id: Optional[Any] = None,
) -> Tuple[str, MessyTextConversationState]:
    conversation_state = state or MessyTextConversationState()
    # ... processing, get result ...
    conversation_state.results.append(result)
    conversation_state.turn_index += 1
    return summary, conversation_state
```

3.1.2 Result output_format:
```json
{
    "info_found": "<TRUE|FALSE>",
    "relevant_context": ["<label keys>"],
    "summary_by_item": {
        "<label_key>": [
            {"span": "<exact text>", "doc_id": "<id>"}
        ]
    },
    "summary": "<combined text>"
}
```

3.1.3 offset (post-hoc):
```python
for label, spans in summary_by_item.items():
    for item in spans:
        source = get_source_text(item['doc_id'])
        item['offset'] = source.find(item['span'])
```

## 4. Span Storage: Relational Design

### 4.1 Main Table (existing)
`df_text_by_report_conversation_evaluation.csv`

| index (PK) | victim | text | summary_all_context | {label}_classification | {label}_match |
|------------|--------|------|---------------------|------------------------|---------------|
| Guerrero_Abel A G_1 | Guerrero_Abel A G | "Abel soñaba ser..." | "Abel, un joven..." | Students | 1 |
| Guerrero_Abel A G_2 | Guerrero_Abel A G | "Página no encontrada..." | "Abel, un joven..." | Students | 1 |

### 4.2 Spans Table (proposed)
`spans.csv`

- 分组键：`label_key`（标识是哪条摘要/claim）
- 记录粒度：`span`（每行一条可追溯证据），携带 `label_key`, `span`, `doc_id`, `turn_index`，`offset` 为后处理可选字段（见 3.1.3）

| index (FK) | label_key | span | doc_id | offset | turn_index |
|------------|-----------|------|--------|--------|------------|
| Guerrero_Abel A G_1 | vic_grupo_social | estudiante de ingeniería civil | doc_0 | 156 | 0 |
| Guerrero_Abel A G_1 | perp_tipo1 | policías de Chilpancingo | doc_0 | 512 | 0 |
| Guerrero_Abel A G_2 | vic_grupo_social | joven estudiante | doc_1 | 89 | 1 |



