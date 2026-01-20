1. seperate the labels into:
the noun class
the verb class
etc...

to answer "what is that one here" question.

like, classify the categories themselves. so others can see it easier.

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
