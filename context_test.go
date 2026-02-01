package vega

import (
	"testing"
)

func TestTokenBudgetContext_Add(t *testing.T) {
	// Create context with small budget (~100 tokens = ~400 chars)
	ctx := NewTokenBudgetContext(100)

	// Add some messages
	ctx.Add(Message{Role: RoleUser, Content: "Hello, how are you?"})
	ctx.Add(Message{Role: RoleAssistant, Content: "I'm doing well, thanks!"})

	msgs := ctx.Messages(0)
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages, got %d", len(msgs))
	}
}

func TestTokenBudgetContext_TrimsOldest(t *testing.T) {
	// Create context with very small budget (~25 tokens = ~100 chars)
	ctx := NewTokenBudgetContext(25)

	// Add messages that exceed budget
	ctx.Add(Message{Role: RoleUser, Content: "This is message one with some text"})       // ~9 tokens
	ctx.Add(Message{Role: RoleAssistant, Content: "This is message two with some text"})  // ~9 tokens
	ctx.Add(Message{Role: RoleUser, Content: "This is message three with some text"})     // ~10 tokens

	msgs := ctx.Messages(0)

	// Should have trimmed the oldest message(s) to stay under budget
	if len(msgs) > 2 {
		t.Errorf("expected at most 2 messages after trimming, got %d", len(msgs))
	}

	// The most recent message should always be kept
	if len(msgs) > 0 && msgs[len(msgs)-1].Content != "This is message three with some text" {
		t.Errorf("expected newest message to be retained")
	}
}

func TestTokenBudgetContext_MessagesRespectsMaxTokens(t *testing.T) {
	ctx := NewTokenBudgetContext(1000) // Large budget

	// Add several messages with known sizes
	// Each message is ~50 chars = ~13 tokens
	ctx.Add(Message{Role: RoleUser, Content: "This is the first message with enough content here"})
	ctx.Add(Message{Role: RoleAssistant, Content: "This is second message with enough content here"})
	ctx.Add(Message{Role: RoleUser, Content: "This is the third message with enough content here"})

	// Request with very small token limit (only 1 message worth)
	msgs := ctx.Messages(15)

	// Should return only the newest message that fits
	if len(msgs) != 1 {
		t.Errorf("expected 1 message with small token limit, got %d", len(msgs))
	}

	// Should be the newest message
	if len(msgs) > 0 && msgs[0].Content != "This is the third message with enough content here" {
		t.Errorf("expected newest message to be returned")
	}
}

func TestTokenBudgetContext_Clear(t *testing.T) {
	ctx := NewTokenBudgetContext(1000)

	ctx.Add(Message{Role: RoleUser, Content: "Hello"})
	ctx.Add(Message{Role: RoleAssistant, Content: "Hi there"})

	ctx.Clear()

	if ctx.TokenCount() != 0 {
		t.Errorf("expected token count 0 after clear, got %d", ctx.TokenCount())
	}

	msgs := ctx.Messages(0)
	if len(msgs) != 0 {
		t.Errorf("expected 0 messages after clear, got %d", len(msgs))
	}
}

func TestTokenBudgetContext_Load(t *testing.T) {
	ctx := NewTokenBudgetContext(1000)

	// Load existing messages
	existing := []Message{
		{Role: RoleUser, Content: "Previous question"},
		{Role: RoleAssistant, Content: "Previous answer"},
	}
	ctx.Load(existing)

	msgs := ctx.Messages(0)
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages after load, got %d", len(msgs))
	}

	if msgs[0].Content != "Previous question" {
		t.Errorf("expected first message to be 'Previous question', got %s", msgs[0].Content)
	}
}

func TestTokenBudgetContext_LoadTrimsIfOverBudget(t *testing.T) {
	// Each message is ~8 tokens (~30 chars), so budget of 20 tokens should keep ~2 messages
	ctx := NewTokenBudgetContext(20)

	// Load messages that exceed budget
	existing := []Message{
		{Role: RoleUser, Content: "This is a long first message!"},
		{Role: RoleAssistant, Content: "This is long second message!"},
		{Role: RoleUser, Content: "This is a long third message!"},
	}
	ctx.Load(existing)

	msgs := ctx.Messages(0)

	// Should have trimmed oldest messages
	if len(msgs) > 2 {
		t.Errorf("expected at most 2 messages after load with trimming, got %d", len(msgs))
	}

	// Most recent should be kept
	if len(msgs) > 0 && msgs[len(msgs)-1].Content != "This is a long third message!" {
		t.Errorf("expected newest message to be retained after load")
	}
}

func TestTokenBudgetContext_Snapshot(t *testing.T) {
	ctx := NewTokenBudgetContext(1000)

	ctx.Add(Message{Role: RoleUser, Content: "Hello"})
	ctx.Add(Message{Role: RoleAssistant, Content: "Hi"})

	snapshot := ctx.Snapshot()

	if len(snapshot) != 2 {
		t.Errorf("expected 2 messages in snapshot, got %d", len(snapshot))
	}

	// Modify snapshot and verify original is unchanged
	snapshot[0].Content = "Modified"

	msgs := ctx.Messages(0)
	if msgs[0].Content == "Modified" {
		t.Error("snapshot modification affected original context")
	}
}

func TestTokenBudgetContext_TokenCount(t *testing.T) {
	ctx := NewTokenBudgetContext(1000)

	if ctx.TokenCount() != 0 {
		t.Errorf("expected initial token count 0, got %d", ctx.TokenCount())
	}

	// "Hello" = 5 chars = ~2 tokens (rounded up: (5+3)/4 = 2)
	ctx.Add(Message{Role: RoleUser, Content: "Hello"})

	if ctx.TokenCount() != 2 {
		t.Errorf("expected token count 2, got %d", ctx.TokenCount())
	}
}

func TestMarshalUnmarshalMessages(t *testing.T) {
	original := []Message{
		{Role: RoleUser, Content: "Hello"},
		{Role: RoleAssistant, Content: "Hi there"},
		{Role: RoleUser, Content: "How are you?"},
	}

	// Marshal
	data, err := MarshalMessages(original)
	if err != nil {
		t.Fatalf("MarshalMessages failed: %v", err)
	}

	// Unmarshal
	restored, err := UnmarshalMessages(data)
	if err != nil {
		t.Fatalf("UnmarshalMessages failed: %v", err)
	}

	if len(restored) != len(original) {
		t.Errorf("expected %d messages, got %d", len(original), len(restored))
	}

	for i := range original {
		if restored[i].Role != original[i].Role {
			t.Errorf("message %d: expected role %s, got %s", i, original[i].Role, restored[i].Role)
		}
		if restored[i].Content != original[i].Content {
			t.Errorf("message %d: expected content %s, got %s", i, original[i].Content, restored[i].Content)
		}
	}
}

func TestUnmarshalMessages_EmptySlice(t *testing.T) {
	data := []byte("[]")
	msgs, err := UnmarshalMessages(data)
	if err != nil {
		t.Fatalf("UnmarshalMessages failed: %v", err)
	}
	if len(msgs) != 0 {
		t.Errorf("expected 0 messages, got %d", len(msgs))
	}
}

func TestUnmarshalMessages_InvalidJSON(t *testing.T) {
	data := []byte("invalid json")
	_, err := UnmarshalMessages(data)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestTokenBudgetContext_ImplementsContextManager(t *testing.T) {
	// Compile-time check that TokenBudgetContext implements ContextManager
	var _ ContextManager = (*TokenBudgetContext)(nil)
}
