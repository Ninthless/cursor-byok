//! Decides when to compact provider-visible context and builds a stable fallback summary.

use std::collections::HashSet;

use crate::{
    model::{
        estimate_context_tokens, estimate_projected_messages_tokens, project_messages,
        CanonicalMessage, ContentPart, MessageContent, Origin, PreparedRun, ProjectedContent,
        ProjectedMessage, Role,
    },
    store::ContextUsageAnchor,
};

const FALLBACK_CHARS: usize = 12_000;

/// Fraction of the context window kept free, as a divisor: 10 = 10%.
///
/// The estimate runs behind the provider: the anchor is what the provider
/// charged for the *previous* call, and the next request re-sends request
/// context and carries provider-side overhead the message-tail estimate does
/// not model. A real conversation measured 948K estimated against 1,017,628
/// actual, a 7% shortfall that landed it over a 1M window while the check said
/// there was room. A proportional reserve absorbs that drift and scales with
/// the model: a 200K window keeps 20K free and a 1M window keeps 100K.
const CONTEXT_RESERVE_DIVISOR: u64 = 10;
pub(super) const OUTPUT_TOKENS: u64 = 4_096;
pub(super) const INSTRUCTIONS: &str = "Summarize the conversation for the next model turn. Preserve goals, constraints, decisions, files, commands, errors, results, and unfinished work. Do not call tools. Return only the concise durable summary.";

/// Usable prompt budget: the window minus the proportional reserve.
pub(super) fn context_budget(context_window: u64) -> u64 {
    context_window.saturating_sub(context_window / CONTEXT_RESERVE_DIVISOR)
}

pub(super) fn input_budget(prepared: &PreparedRun) -> Option<u64> {
    prepared.model.context_window_tokens.map(context_budget)
}

/// Whether a provider failure means the prompt did not fit.
///
/// Providers report this as a plain 400 with prose, so there is nothing
/// structured to match on. Anthropic says "prompt is too long"; OpenAI-style
/// gateways use `context_length_exceeded` or "maximum context length".
pub(super) fn is_context_overflow(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("prompt is too long")
        || lowered.contains("context window exceeded")
        || lowered.contains("model_context_window_exceeded")
        || lowered.contains("context_length_exceeded")
        || (lowered.contains("maximum context length") && lowered.contains("token"))
}

/// Builds the history for a compaction call.
///
/// Two properties matter, and replaying the raw history guarantees neither:
///
/// 1. The history must end with a user message. Otherwise providers read the
///    request as an assistant prefill and refuse it outright: Anthropic answers
///    "This model does not support assistant message prefill. The conversation
///    must end with a user message."
/// 2. The history must fit the context window. Compaction runs precisely
///    because the conversation is too large, so replaying all of it asks the
///    summarizer to accept a prompt that is already over the limit and the
///    call fails with "prompt is too long".
///
/// Either failure falls back to the truncated summary, which is usually still
/// too large, so the conversation stays over its window and cannot recover.
///
/// Trimming keeps the oldest turns of the dropped prefix and only ever cuts at
/// a user-message boundary, so an assistant tool call is never separated from
/// its results. Recent turns belong in `window_tail`, not in the summarizer.
pub(super) fn compaction_history(
    history: Vec<ProjectedMessage>,
    context_window: Option<u64>,
) -> Vec<ProjectedMessage> {
    super::history::user_terminated(
        trim_to_context(history, context_window),
        "compaction:instruction",
        INSTRUCTIONS,
    )
}

fn trim_to_context(
    mut history: Vec<ProjectedMessage>,
    context_window: Option<u64>,
) -> Vec<ProjectedMessage> {
    let Some(budget) = context_window
        .filter(|window| *window > 0)
        .map(context_budget)
        .map(|budget| budget.saturating_sub(OUTPUT_TOKENS))
        .filter(|budget| *budget > 0)
    else {
        return history;
    };
    if estimate_projected_messages_tokens(&history) <= budget {
        return history;
    }
    // Walk forward from the oldest turn, keeping whole user-delimited turns.
    let mut end = 0;
    for (index, message) in history.iter().enumerate() {
        if !is_turn_boundary(message) {
            continue;
        }
        if index > 0 && estimate_projected_messages_tokens(&history[..index]) > budget {
            break;
        }
        if index > 0 {
            end = index;
        }
    }
    if end == 0 {
        // Not even the oldest turn fits. Keep it anyway rather than sending an
        // empty history: an empty summarize call returns a summary of nothing
        // that would then replace the whole conversation.
        let next = history
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, message)| is_turn_boundary(message))
            .map(|(index, _)| index)
            .unwrap_or(history.len());
        history.truncate(next);
        return history;
    }
    history.truncate(end);
    history
}

fn is_turn_boundary(message: &ProjectedMessage) -> bool {
    message.role == Role::User && matches!(message.content, ProjectedContent::Parts(_))
}

pub(super) fn estimated_tokens(
    prepared: &PreparedRun,
    projected_messages: &[ProjectedMessage],
    anchor: Option<ContextUsageAnchor>,
) -> u64 {
    anchor
        .filter(|anchor| anchor.message_count <= projected_messages.len())
        .map(|anchor| {
            anchor
                .context_input_tokens
                .saturating_add(estimate_projected_messages_tokens(
                    &projected_messages[anchor.message_count..],
                ))
        })
        .unwrap_or_else(|| estimate_context_tokens(&prepared.prompt, projected_messages))
}

pub(super) fn compaction_estimate(
    prepared: &PreparedRun,
    projected_messages: &[ProjectedMessage],
    anchor: Option<ContextUsageAnchor>,
) -> Option<u64> {
    let budget = input_budget(prepared)?;
    let estimated = estimated_tokens(prepared, projected_messages, anchor);
    (estimated > budget).then_some(estimated)
}

#[cfg(test)]
pub(super) fn should_compact(
    prepared: &PreparedRun,
    projected_messages: &[ProjectedMessage],
    anchor: Option<ContextUsageAnchor>,
) -> bool {
    compaction_estimate(prepared, projected_messages, anchor).is_some()
}

pub(super) fn validate_compacted(
    prepared: &PreparedRun,
    projected_messages: &[ProjectedMessage],
) -> std::result::Result<u64, String> {
    let estimated = estimate_context_tokens(&prepared.prompt, projected_messages);
    let Some(budget) = input_budget(prepared) else {
        return Ok(estimated);
    };
    if estimated <= budget {
        return Ok(estimated);
    }
    Err(format!(
        "context overflow after compaction: estimated input {estimated} tokens exceeds budget {budget} tokens"
    ))
}

pub(super) fn partition(
    messages: &[CanonicalMessage],
    current_ids: &HashSet<&str>,
) -> (Vec<CanonicalMessage>, Option<CanonicalMessage>) {
    let latest_request_context = messages
        .iter()
        .rposition(|message| message.message_id.starts_with("request-context:"));
    let compactable = messages
        .iter()
        .enumerate()
        .filter(|(index, message)| {
            Some(*index) != latest_request_context
                && !current_ids.contains(message.message_id.as_str())
        })
        .map(|(_, message)| message.clone())
        .collect();
    let retained = latest_request_context
        .and_then(|index| messages.get(index))
        .filter(|message| !current_ids.contains(message.message_id.as_str()))
        .cloned();
    (compactable, retained)
}

#[derive(Clone, Copy)]
pub(super) enum CompactionKind {
    Auto,
    Manual,
}

/// Cursor-style window: summarize the obsolete prefix, keep a recent tail.
///
/// Local Cursor 3.20.17 agent-host `partitionMessages` keeps the last user
/// turn as `preservedTailMessages` and puts earlier turns (file bubbles, Read
/// results) in `messagesToSummarize`. Composer then stops resending those
/// bubbles at `truncation_last_bubble_id_inclusive`. BYOK inlines the same
/// files into `<selected_context>` / Read results, so a token-budget tail
/// would still carry them; the first pass peels that file window into the
/// prefix instead of leaving it in `window_tail` or current initial.
pub(super) struct CompactionPlan {
    pub prefix: Vec<CanonicalMessage>,
    pub tail: Vec<CanonicalMessage>,
    pub retained_request_context: Option<CanonicalMessage>,
}

impl CompactionPlan {
    pub(super) fn window_tail(&self) -> u32 {
        self.tail.len().min(u32::MAX as usize) as u32
    }
}

pub(super) fn plan(
    messages: &[CanonicalMessage],
    current_ids: &HashSet<&str>,
    prepared: &PreparedRun,
    kind: CompactionKind,
) -> Option<CompactionPlan> {
    let (compactable, retained_request_context) = partition(messages, current_ids);
    let current_file_windows = prepared
        .initial_messages
        .iter()
        .filter(|message| is_file_window(message))
        .cloned()
        .collect::<Vec<_>>();
    if compactable.is_empty() && current_file_windows.is_empty() {
        return None;
    }
    let tail_start = tail_start_index(
        &compactable,
        tail_token_budget(prepared, retained_request_context.as_ref()),
    );
    let compactable_has_file_window = compactable.iter().any(is_file_window);
    let (mut prefix, mut tail) = if compactable.is_empty() {
        (Vec::new(), Vec::new())
    } else if tail_start == 0 {
        match kind {
            CompactionKind::Auto
                if current_file_windows.is_empty() && !compactable_has_file_window =>
            {
                return None;
            }
            CompactionKind::Auto => (Vec::new(), compactable),
            CompactionKind::Manual => (compactable, Vec::new()),
        }
    } else {
        let mut compactable = compactable;
        let tail = compactable.split_off(tail_start);
        (compactable, tail)
    };
    peel_file_windows(&mut prefix, &mut tail);
    prefix.extend(current_file_windows);
    if prefix.is_empty() {
        return None;
    }
    Some(CompactionPlan {
        prefix,
        tail,
        retained_request_context,
    })
}

pub(super) fn summary_message(event_id: String, summary: &str) -> CanonicalMessage {
    CanonicalMessage {
        message_id: format!("runtime:{event_id}"),
        role: Role::User,
        origin: Origin::Runtime,
        content: MessageContent::Parts {
            parts: vec![ContentPart::Text {
                text: format!("<conversation_summary>\n{summary}\n</conversation_summary>"),
            }],
        },
        runtime_event_id: Some(event_id),
    }
}

pub(super) fn replacement(
    plan: &CompactionPlan,
    summary: CanonicalMessage,
    initial: &[CanonicalMessage],
) -> Vec<CanonicalMessage> {
    let mut messages = Vec::with_capacity(
        usize::from(plan.retained_request_context.is_some()) + 1 + plan.tail.len() + initial.len(),
    );
    if let Some(context) = &plan.retained_request_context {
        messages.push(context.clone());
    }
    messages.push(summary);
    messages.extend(plan.tail.iter().cloned());
    messages.extend(initial.iter().cloned().map(|message| {
        if is_file_window(&message) {
            shrink_file_window(message)
        } else {
            message
        }
    }));
    messages
}

const COMPACTED_FILE_WINDOW: &str = "[compacted file window]";
const COMPACTED_IDENTITY_SUFFIX: &str = ":compacted-file-window";

fn is_file_window(message: &CanonicalMessage) -> bool {
    match &message.content {
        MessageContent::Parts { parts } => parts.iter().any(|part| match part {
            ContentPart::Text { text } => contains_selected_file_window(text),
            _ => false,
        }),
        MessageContent::ToolResult(result) => {
            result.name == "Read"
                && !result.content.is_empty()
                && result.content != COMPACTED_FILE_WINDOW
        }
        MessageContent::Assistant { .. } => false,
    }
}

fn contains_selected_file_window(text: &str) -> bool {
    let Some(start) = text.find("<selected_context>") else {
        return false;
    };
    let Some(relative_end) = text[start..].find("</selected_context>") else {
        return false;
    };
    let block = &text[start..start + relative_end];
    block.contains("<file") || block.contains("<code ")
}

fn peel_file_windows(prefix: &mut Vec<CanonicalMessage>, tail: &mut Vec<CanonicalMessage>) {
    let mut kept = Vec::with_capacity(tail.len());
    for message in tail.drain(..) {
        if is_file_window(&message) {
            prefix.push(message.clone());
            kept.push(shrink_file_window(message));
        } else {
            kept.push(message);
        }
    }
    *tail = kept;
}

fn shrink_file_window(mut message: CanonicalMessage) -> CanonicalMessage {
    match &mut message.content {
        MessageContent::Parts { parts } => {
            for part in parts {
                if let ContentPart::Text { text } = part {
                    *text = strip_selected_context(text);
                }
            }
        }
        MessageContent::ToolResult(result) if result.name == "Read" => {
            result.content = COMPACTED_FILE_WINDOW.into();
        }
        _ => {}
    }
    // Checkpoint messages are immutable by id. Keep the shrunk body on a new
    // identity so replace_checkpoint does not hit "reused with different content".
    retarget_compacted_identity(&mut message);
    message
}

fn retarget_compacted_identity(message: &mut CanonicalMessage) {
    if !message.message_id.ends_with(COMPACTED_IDENTITY_SUFFIX) {
        message.message_id.push_str(COMPACTED_IDENTITY_SUFFIX);
    }
    if let Some(event_id) = &mut message.runtime_event_id {
        if !event_id.ends_with(COMPACTED_IDENTITY_SUFFIX) {
            event_id.push_str(COMPACTED_IDENTITY_SUFFIX);
        }
    }
}

fn strip_selected_context(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::new();
    while let Some(start) = remaining.find("<selected_context>") {
        output.push_str(remaining[..start].trim_end());
        let Some(relative_end) = remaining[start..].find("</selected_context>") else {
            output.push_str(remaining[start..].trim_start());
            return output;
        };
        remaining = remaining[start + relative_end + "</selected_context>".len()..].trim_start();
        if !output.is_empty() && !remaining.is_empty() {
            output.push_str("\n\n");
        }
    }
    output.push_str(remaining);
    output
}

fn tail_token_budget(
    prepared: &PreparedRun,
    retained_request_context: Option<&CanonicalMessage>,
) -> Option<u64> {
    let budget = input_budget(prepared)?;
    let mut held = Vec::new();
    if let Some(context) = retained_request_context {
        held.push(context.clone());
    }
    held.extend(prepared.initial_messages.iter().cloned());
    let held_tokens = project_messages(&held)
        .map(|projected| estimate_context_tokens(&prepared.prompt, &projected))
        .unwrap_or(0);
    Some(
        budget
            .saturating_sub(held_tokens)
            .saturating_sub(OUTPUT_TOKENS),
    )
}

fn tail_start_index(compactable: &[CanonicalMessage], tail_budget: Option<u64>) -> usize {
    if compactable.is_empty() {
        return 0;
    }
    let Some(budget) = tail_budget else {
        return last_turn_start(compactable);
    };
    if budget == 0 {
        return compactable.len();
    }
    let Ok(projected) = project_messages(compactable) else {
        return compactable.len();
    };
    let mut start = compactable.len();
    for (index, message) in compactable.iter().enumerate().rev() {
        if !is_canonical_turn_boundary(message) {
            continue;
        }
        if estimate_projected_messages_tokens(&projected[index..]) > budget {
            break;
        }
        start = index;
    }
    start
}

fn last_turn_start(compactable: &[CanonicalMessage]) -> usize {
    compactable
        .iter()
        .rposition(is_canonical_turn_boundary)
        .unwrap_or(0)
}

fn is_canonical_turn_boundary(message: &CanonicalMessage) -> bool {
    message.role == Role::User && matches!(message.content, MessageContent::Parts { .. })
}

pub(super) fn fallback_summary(messages: &[CanonicalMessage]) -> String {
    let serialized = serde_json::to_string(messages).unwrap_or_default();
    let end = serialized
        .char_indices()
        .nth(FALLBACK_CHARS.saturating_sub(1))
        .map_or(serialized.len(), |(index, _)| index);
    format!("Durable conversation state:\n{}", &serialized[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        project_messages, CheckpointId, ContentPart, ConversationId, ModelSpec, Origin, PromptSpec,
        Role, RunAction, RunId, RunKind,
    };

    fn prepared(context_window_tokens: u64) -> PreparedRun {
        let mut model = ModelSpec::new("model");
        model.context_window_tokens = Some(context_window_tokens);
        PreparedRun {
            run_id: RunId::new("run"),
            cursor_request_id: None,
            conversation_id: ConversationId::new("conversation"),
            kind: RunKind::Root,
            model,
            prompt: PromptSpec {
                instructions: String::new(),
                tools: Vec::new(),
            },
            initial_messages: Vec::new(),
            action: RunAction::Start,
            base_checkpoint_id: CheckpointId(1),
        }
    }

    #[test]
    fn automatic_compaction_uses_proportional_reserve_for_every_action() {
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "x".repeat(40_000),
        )];
        let projected = project_messages(&messages).unwrap();
        let estimated = estimate_context_tokens(&prepared(1).prompt, &projected);
        // Smallest multiple-of-ten window whose 90% budget covers the estimate.
        let window = estimated.div_ceil(9) * 10;
        let mut prepared = prepared(window);
        assert!(context_budget(window) >= estimated);
        assert!(context_budget(window - 10) < estimated);

        assert!(!should_compact(&prepared, &projected, None));
        prepared.model.context_window_tokens = Some(window - 10);
        assert!(should_compact(&prepared, &projected, None));

        prepared.action = RunAction::Resume {
            pending_tool_round: None,
        };
        assert!(should_compact(&prepared, &projected, None));
    }

    #[test]
    fn the_reserve_leaves_room_for_the_estimate_to_run_behind() {
        // Reproduces a conversation that wedged itself against a 1M window.
        // The anchor said 947,797 tokens, so a fixed 10K reserve found room
        // and let the request through. Anthropic counted 1,017,628 and
        // refused it, and every retry repeated the same arithmetic.
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "hello",
        )];
        let projected = project_messages(&messages).unwrap();
        let anchor = ContextUsageAnchor {
            context_input_tokens: 947_797,
            message_count: 1,
        };
        assert!(should_compact(
            &prepared(1_000_000),
            &projected,
            Some(anchor)
        ));
    }

    #[test]
    fn the_reserve_scales_with_the_window() {
        assert_eq!(context_budget(200_000), 180_000);
        assert_eq!(context_budget(1_000_000), 900_000);
        assert_eq!(context_budget(0), 0);
        assert_eq!(context_budget(1), 1);
    }

    #[test]
    fn provider_refusals_that_mean_the_prompt_did_not_fit_are_recognized() {
        assert!(is_context_overflow(
            "provider error: Anthropic 400 Bad Request: {\"type\":\"error\",\"error\":\
             {\"type\":\"invalid_request_error\",\"message\":\"prompt is too long: \
             1017628 tokens > 1000000 maximum\"}}"
        ));
        assert!(is_context_overflow("model_context_window_exceeded"));
        assert!(is_context_overflow("context_length_exceeded"));
        assert!(is_context_overflow(
            "This model's maximum context length is 128000 tokens"
        ));

        // Unrelated failures must not trigger a compaction, which would
        // destroy history to fix something compaction cannot fix.
        assert!(!is_context_overflow("401 Unauthorized: invalid api key"));
        assert!(!is_context_overflow("429 Too Many Requests"));
        assert!(!is_context_overflow(
            "This model does not support assistant message prefill"
        ));
    }

    fn user(id: &str, text: &str) -> ProjectedMessage {
        ProjectedMessage {
            message_id: id.into(),
            role: Role::User,
            content: ProjectedContent::Parts(vec![ContentPart::Text { text: text.into() }]),
        }
    }

    fn assistant(id: &str, text: &str) -> ProjectedMessage {
        ProjectedMessage {
            message_id: id.into(),
            role: Role::Assistant,
            content: ProjectedContent::Assistant {
                text: text.into(),
                thinking: String::new(),
                replay_state: None,
                calls: Vec::new(),
            },
        }
    }

    #[test]
    fn compaction_history_always_ends_with_a_user_message() {
        // Providers reject an assistant-terminated history as a prefill, which
        // made every automatic compaction fall back to the truncated summary.
        let history = vec![user("u1", "question"), assistant("a1", "answer")];
        let prepared = compaction_history(history, Some(200_000));
        assert_eq!(prepared.last().unwrap().role, Role::User);
        assert_eq!(
            prepared.last().unwrap().message_id,
            "compaction:instruction"
        );

        // An already user-terminated history is left alone.
        let history = vec![assistant("a1", "answer"), user("u2", "next")];
        let prepared = compaction_history(history.clone(), Some(200_000));
        assert_eq!(prepared, history);
    }

    #[test]
    fn compaction_history_is_trimmed_to_fit_the_context_window() {
        // The summarizer sees the dropped prefix. If that prefix is itself over
        // the window, keep the oldest turns: goals live there, and the newest
        // turns are already retained as window_tail.
        let big = "x".repeat(400_000);
        let history = vec![
            user("u1", &big),
            assistant("a1", &big),
            user("u2", &big),
            assistant("a2", "recent answer"),
        ];
        let window = 200_000;
        let prepared = compaction_history(history, Some(window));

        let budget = context_budget(window) - OUTPUT_TOKENS;
        assert!(
            estimate_projected_messages_tokens(&prepared) <= budget
                || prepared.iter().any(|message| message.message_id == "u1")
        );
        assert_eq!(prepared.last().unwrap().role, Role::User);
        assert!(prepared.iter().any(|message| message.message_id == "u1"));
        assert!(!prepared.iter().any(|message| message.message_id == "u2"));
    }

    #[test]
    fn compaction_history_keeps_the_oldest_turn_even_when_it_is_over_budget() {
        // A prefix whose oldest turn alone exceeds the budget is still sent
        // rather than trimmed to nothing: a summary of nothing would replace
        // the whole conversation.
        let history = vec![
            user("u1", &"x".repeat(400_000)),
            assistant("a1", "old answer"),
            user("u2", "recent"),
            assistant("a2", "answer"),
        ];
        let prepared = compaction_history(history, Some(50_000));
        assert_eq!(prepared[0].message_id, "u1");
        assert!(!prepared.iter().any(|message| message.message_id == "u2"));
        assert_eq!(prepared.last().unwrap().role, Role::User);
    }

    fn ids(messages: &[CanonicalMessage]) -> HashSet<&str> {
        messages
            .iter()
            .map(|message| message.message_id.as_str())
            .collect()
    }

    #[test]
    fn auto_plan_keeps_a_recent_turn_as_window_tail() {
        let old = CanonicalMessage::text("u1", Role::User, Origin::Runtime, "x".repeat(200_000));
        let old_answer = CanonicalMessage::text("a1", Role::Assistant, Origin::Assistant, "old");
        let recent = CanonicalMessage::text("u2", Role::User, Origin::Runtime, "recent work");
        let recent_answer =
            CanonicalMessage::text("a2", Role::Assistant, Origin::Assistant, "done");
        let current = CanonicalMessage::text("u3", Role::User, Origin::Runtime, "continue");
        let messages = vec![old, old_answer, recent, recent_answer, current.clone()];
        let current_ids = ids(std::slice::from_ref(&current));
        let mut prepared = prepared(50_000);
        prepared.initial_messages = vec![current.clone()];

        let plan = plan(&messages, &current_ids, &prepared, CompactionKind::Auto).unwrap();
        assert_eq!(
            plan.prefix
                .iter()
                .map(|message| message.message_id.as_str())
                .collect::<Vec<_>>(),
            ["u1", "a1"]
        );
        assert_eq!(
            plan.tail
                .iter()
                .map(|message| message.message_id.as_str())
                .collect::<Vec<_>>(),
            ["u2", "a2"]
        );
        assert_eq!(plan.window_tail(), 2);
        let assembled = replacement(&plan, summary_message("s".into(), "goals and files"), &[]);
        assert_eq!(assembled[0].message_id, "runtime:s");
        assert_eq!(assembled[1].message_id, "u2");
    }

    #[test]
    fn auto_plan_skips_when_the_compactable_dialogue_already_fits_in_the_tail() {
        let context = CanonicalMessage::text(
            "request-context:rules",
            Role::User,
            Origin::Runtime,
            "x".repeat(80_000),
        );
        let user = CanonicalMessage::text("u1", Role::User, Origin::Runtime, "hi");
        let assistant = CanonicalMessage::text("a1", Role::Assistant, Origin::Assistant, "hello");
        let current = CanonicalMessage::text("u2", Role::User, Origin::Runtime, "again");
        let messages = vec![context, user, assistant, current.clone()];
        let current_ids = ids(std::slice::from_ref(&current));
        let mut prepared = prepared(200_000);
        prepared.initial_messages = vec![current.clone()];

        assert!(plan(&messages, &current_ids, &prepared, CompactionKind::Auto).is_none());
    }

    fn file_window_user(id: &str, query: &str, body: &str) -> CanonicalMessage {
        CanonicalMessage::text(
            id,
            Role::User,
            Origin::Runtime,
            format!(
                "<selected_context>\n<file path=\"src/big.rs\">\n{body}\n</file>\n</selected_context>\n\n<user_query>\n{query}\n</user_query>"
            ),
        )
    }

    #[test]
    fn auto_plan_moves_selected_context_file_window_into_the_prefix() {
        // Cursor 3.20.17 drops file-bearing bubbles via
        // truncation_last_bubble_id_inclusive. A token-budget tail would keep
        // this dump because it still fits, which is the 96% first-pass leftover.
        let old = CanonicalMessage::text("u1", Role::User, Origin::Runtime, "x".repeat(200_000));
        let old_answer = CanonicalMessage::text("a1", Role::Assistant, Origin::Assistant, "old");
        let files = file_window_user("u2", "recent work", &"secret-file-body".repeat(8_000));
        let files_answer =
            CanonicalMessage::text("a2", Role::Assistant, Origin::Assistant, "noted");
        let current = CanonicalMessage::text("u3", Role::User, Origin::Runtime, "continue");
        let messages = vec![old, old_answer, files, files_answer, current.clone()];
        let current_ids = ids(std::slice::from_ref(&current));
        let mut prepared = prepared(200_000);
        prepared.initial_messages = vec![current.clone()];

        let plan = plan(&messages, &current_ids, &prepared, CompactionKind::Auto).unwrap();
        assert!(
            plan.prefix.iter().any(|message| {
                message.message_id == "u2" && message_text(message).contains("secret-file-body")
            }),
            "summarizer must see the file window"
        );
        assert!(
            plan.tail.iter().any(|message| {
                message.message_id != "u2" && message_text(message).contains("recent work")
            }),
            "recent query stays in the tail under a new identity"
        );
        assert!(
            plan.tail
                .iter()
                .all(|message| !message_text(message).contains("secret-file-body")),
            "tail must not keep file bodies"
        );
        let assembled = replacement(
            &plan,
            summary_message("s".into(), "files were summarized"),
            &prepared.initial_messages,
        );
        assert!(assembled
            .iter()
            .all(|message| !message_text(message).contains("secret-file-body")));
        assert!(assembled.iter().all(|message| message.message_id != "u2"));
        assert!(assembled.iter().any(|message| message.message_id == "u3"));
    }

    #[test]
    fn replacement_strips_file_window_from_current_initial() {
        let old = CanonicalMessage::text("u1", Role::User, Origin::Runtime, "x".repeat(200_000));
        let old_answer = CanonicalMessage::text("a1", Role::Assistant, Origin::Assistant, "old");
        let current = file_window_user("u2", "continue", &"current-file-body".repeat(8_000));
        let messages = vec![old, old_answer, current.clone()];
        let current_ids = ids(std::slice::from_ref(&current));
        let mut prepared = prepared(50_000);
        prepared.initial_messages = vec![current.clone()];

        let plan = plan(&messages, &current_ids, &prepared, CompactionKind::Auto).unwrap();
        assert!(plan.prefix.iter().any(|message| {
            message.message_id == "u2" && message_text(message).contains("current-file-body")
        }));
        let assembled = replacement(
            &plan,
            summary_message("s".into(), "current files summarized"),
            &prepared.initial_messages,
        );
        let slim = assembled
            .iter()
            .find(|message| message_text(message).contains("continue"))
            .unwrap();
        assert_ne!(slim.message_id, "u2");
        assert!(message_text(slim).contains("continue"));
        assert!(!message_text(slim).contains("current-file-body"));
        assert!(!message_text(slim).contains("<selected_context>"));
    }

    #[test]
    fn shrinking_file_windows_does_not_reuse_a_persisted_identity() {
        let mut current = file_window_user(
            "cursor-root:N5ZiHeATMhBKSyUyDvOkWZ2Br3ygG+guJez/HoE0lqo=:4",
            "continue",
            "current-file-body",
        );
        current.runtime_event_id = Some(current.message_id.clone());
        let original_id = current.message_id.clone();
        let original_event = current.runtime_event_id.clone();
        let old = CanonicalMessage::text("u1", Role::User, Origin::Runtime, "x".repeat(200_000));
        let old_answer = CanonicalMessage::text("a1", Role::Assistant, Origin::Assistant, "old");
        let messages = vec![old, old_answer, current.clone()];
        let current_ids = ids(std::slice::from_ref(&current));
        let mut prepared = prepared(50_000);
        prepared.initial_messages = vec![current.clone()];

        let plan = plan(&messages, &current_ids, &prepared, CompactionKind::Auto).unwrap();
        let assembled = replacement(
            &plan,
            summary_message("s".into(), "current files summarized"),
            &prepared.initial_messages,
        );
        for message in &assembled {
            let same_id =
                message.message_id == original_id || message.runtime_event_id == original_event;
            assert!(
                !same_id,
                "shrunk replacement reused {}:{}",
                message.message_id,
                message.runtime_event_id.as_deref().unwrap_or("-")
            );
        }
        assert!(assembled.iter().any(|message| {
            message.message_id.ends_with(COMPACTED_IDENTITY_SUFFIX)
                && message_text(message).contains("continue")
        }));
    }

    fn message_text(message: &CanonicalMessage) -> String {
        match &message.content {
            MessageContent::Parts { parts } => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
            MessageContent::ToolResult(result) => result.content.clone(),
            MessageContent::Assistant { text, .. } => text.clone(),
        }
    }

    #[test]
    fn manual_plan_summarizes_a_single_turn_when_there_is_no_tail() {
        let user = CanonicalMessage::text("u1", Role::User, Origin::Runtime, "remember alpha");
        let assistant =
            CanonicalMessage::text("a1", Role::Assistant, Origin::Assistant, "old answer");
        let messages = vec![user, assistant];
        let prepared = prepared(200_000);
        let plan = plan(
            &messages,
            &HashSet::new(),
            &prepared,
            CompactionKind::Manual,
        )
        .unwrap();
        assert_eq!(plan.prefix.len(), 2);
        assert!(plan.tail.is_empty());
        assert_eq!(plan.window_tail(), 0);
    }

    #[test]
    fn fallback_summary_keeps_the_oldest_prefix_bytes() {
        let messages = vec![
            CanonicalMessage::text("u1", Role::User, Origin::Runtime, "alpha-goal"),
            CanonicalMessage::text("u2", Role::User, Origin::Runtime, "z".repeat(20_000)),
        ];
        let fallback = fallback_summary(&messages);
        assert!(fallback.contains("alpha-goal"));
    }

    #[test]
    fn compaction_history_without_a_context_window_is_untouched_apart_from_termination() {
        let history = vec![user("u1", "question"), assistant("a1", "answer")];
        let prepared = compaction_history(history.clone(), None);
        assert_eq!(prepared[..2], history[..]);
        assert_eq!(prepared.last().unwrap().role, Role::User);
    }

    #[test]
    fn provider_usage_anchor_only_estimates_messages_added_after_last_request() {
        let messages = vec![
            CanonicalMessage::text("old", Role::User, Origin::Runtime, "x".repeat(400_000)),
            CanonicalMessage::text("new", Role::User, Origin::Runtime, "short follow-up"),
        ];
        let projected = project_messages(&messages).unwrap();
        let anchor = ContextUsageAnchor {
            context_input_tokens: 103_904,
            message_count: 1,
        };
        let expected = 103_904 + estimate_projected_messages_tokens(&projected[1..]);

        assert_eq!(
            estimated_tokens(&prepared(200_000), &projected, Some(anchor)),
            expected
        );
        assert!(!should_compact(
            &prepared(200_000),
            &projected,
            Some(anchor)
        ));
    }

    #[test]
    fn provider_usage_anchor_triggers_after_new_messages_cross_budget() {
        let messages = vec![
            CanonicalMessage::text("old", Role::User, Origin::Runtime, "old"),
            CanonicalMessage::text("new", Role::User, Origin::Runtime, "x".repeat(80_000)),
        ];
        let projected = project_messages(&messages).unwrap();

        assert!(should_compact(
            &prepared(200_000),
            &projected,
            Some(ContextUsageAnchor {
                context_input_tokens: 180_000,
                message_count: 1,
            })
        ));
    }

    #[test]
    fn missing_anchor_uses_full_fallback() {
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "x".repeat(40_000),
        )];
        let projected = project_messages(&messages).unwrap();
        let prepared = prepared(200_000);

        assert_eq!(
            estimated_tokens(&prepared, &projected, None),
            estimate_context_tokens(&prepared.prompt, &projected)
        );
    }

    #[test]
    fn invalid_anchor_message_count_uses_full_fallback() {
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "x".repeat(40_000),
        )];
        let projected = project_messages(&messages).unwrap();
        let expected = estimate_context_tokens(&prepared(200_000).prompt, &projected);

        assert_eq!(
            estimated_tokens(
                &prepared(200_000),
                &projected,
                Some(ContextUsageAnchor {
                    context_input_tokens: 1,
                    message_count: 2,
                })
            ),
            expected
        );
    }

    #[test]
    fn compacted_history_is_validated_against_the_same_budget() {
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "x".repeat(40_000),
        )];
        let projected = project_messages(&messages).unwrap();
        let estimated = estimate_context_tokens(&prepared(1).prompt, &projected);
        let window = estimated.div_ceil(9) * 10;
        assert!(context_budget(window) >= estimated);
        assert!(context_budget(window - 10) < estimated);

        assert_eq!(
            validate_compacted(&prepared(window), &projected),
            Ok(estimated)
        );
        assert!(validate_compacted(&prepared(window - 10), &projected)
            .unwrap_err()
            .contains("context overflow after compaction"));
    }
}
