package vega

import (
	"encoding/json"
	"sync"
)

// TokenBudgetContext manages conversation history within a token budget.
// When adding messages would exceed the budget, oldest messages are removed.
type TokenBudgetContext struct {
	messages   []Message
	maxTokens  int
	tokenCount int
	mu         sync.RWMutex
}

// NewTokenBudgetContext creates a context that keeps messages within a token budget.
// Uses ~4 chars per token as a rough estimate.
func NewTokenBudgetContext(maxTokens int) *TokenBudgetContext {
	return &TokenBudgetContext{
		messages:  make([]Message, 0),
		maxTokens: maxTokens,
	}
}

// estimateTokens estimates token count for a message (~4 chars per token).
func estimateTokens(content string) int {
	return (len(content) + 3) / 4 // Round up
}

// Add appends a message to the context.
// If the new message would exceed the budget, oldest messages are removed.
func (c *TokenBudgetContext) Add(msg Message) {
	c.mu.Lock()
	defer c.mu.Unlock()

	msgTokens := estimateTokens(msg.Content)
	c.messages = append(c.messages, msg)
	c.tokenCount += msgTokens

	// Trim oldest messages until we're under budget
	for c.tokenCount > c.maxTokens && len(c.messages) > 1 {
		removed := c.messages[0]
		c.messages = c.messages[1:]
		c.tokenCount -= estimateTokens(removed.Content)
	}
}

// Messages returns messages that fit within maxTokens.
// If the requested maxTokens is lower than our budget, we return fewer messages.
func (c *TokenBudgetContext) Messages(maxTokens int) []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Respect the requested maxTokens (may be lower than our budget)
	effectiveMax := c.maxTokens
	if maxTokens > 0 && maxTokens < effectiveMax {
		effectiveMax = maxTokens
	}

	// Return messages from newest to oldest until we hit the limit
	result := make([]Message, 0, len(c.messages))
	tokens := 0
	for i := len(c.messages) - 1; i >= 0; i-- {
		msgTokens := estimateTokens(c.messages[i].Content)
		if tokens+msgTokens > effectiveMax {
			break
		}
		result = append([]Message{c.messages[i]}, result...)
		tokens += msgTokens
	}
	return result
}

// Clear resets the context.
func (c *TokenBudgetContext) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = c.messages[:0]
	c.tokenCount = 0
}

// TokenCount returns current token usage.
func (c *TokenBudgetContext) TokenCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tokenCount
}

// Load initializes the context with existing messages.
// This is useful for restoring conversation history from persistence.
// If the loaded messages exceed the budget, oldest messages are trimmed.
func (c *TokenBudgetContext) Load(messages []Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = make([]Message, 0, len(messages))
	c.tokenCount = 0

	for _, msg := range messages {
		c.messages = append(c.messages, msg)
		c.tokenCount += estimateTokens(msg.Content)
	}

	// Trim if loaded messages exceed budget
	for c.tokenCount > c.maxTokens && len(c.messages) > 1 {
		removed := c.messages[0]
		c.messages = c.messages[1:]
		c.tokenCount -= estimateTokens(removed.Content)
	}
}

// Snapshot returns a copy of all messages for persistence.
func (c *TokenBudgetContext) Snapshot() []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	result := make([]Message, len(c.messages))
	copy(result, c.messages)
	return result
}

// MarshalMessages serializes messages to JSON.
func MarshalMessages(messages []Message) ([]byte, error) {
	return json.Marshal(messages)
}

// UnmarshalMessages deserializes messages from JSON.
func UnmarshalMessages(data []byte) ([]Message, error) {
	var messages []Message
	err := json.Unmarshal(data, &messages)
	return messages, err
}
