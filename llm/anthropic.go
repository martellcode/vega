// Package llm provides LLM backend implementations.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/martellcode/vega"
)

// AnthropicLLM is an LLM implementation using the Anthropic API.
type AnthropicLLM struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	model      string
}

// AnthropicOption configures the Anthropic client.
type AnthropicOption func(*AnthropicLLM)

// WithAPIKey sets the API key.
func WithAPIKey(key string) AnthropicOption {
	return func(a *AnthropicLLM) {
		a.apiKey = key
	}
}

// WithModel sets the default model.
func WithModel(model string) AnthropicOption {
	return func(a *AnthropicLLM) {
		a.model = model
	}
}

// WithBaseURL sets the API base URL.
func WithBaseURL(url string) AnthropicOption {
	return func(a *AnthropicLLM) {
		a.baseURL = url
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) AnthropicOption {
	return func(a *AnthropicLLM) {
		a.httpClient = client
	}
}

// Default Anthropic configuration values
const (
	DefaultAnthropicTimeout = 5 * time.Minute
	DefaultAnthropicModel   = "claude-sonnet-4-20250514"
	DefaultAnthropicBaseURL = "https://api.anthropic.com"
)

// NewAnthropic creates a new Anthropic LLM client.
func NewAnthropic(opts ...AnthropicOption) *AnthropicLLM {
	a := &AnthropicLLM{
		apiKey:  os.Getenv("ANTHROPIC_API_KEY"),
		baseURL: DefaultAnthropicBaseURL,
		httpClient: &http.Client{
			Timeout: DefaultAnthropicTimeout,
		},
		model: DefaultAnthropicModel,
	}

	for _, opt := range opts {
		opt(a)
	}

	return a
}

// anthropicRequest is the API request format.
type anthropicRequest struct {
	Model       string           `json:"model"`
	Messages    []anthropicMsg   `json:"messages"`
	System      string           `json:"system,omitempty"`
	MaxTokens   int              `json:"max_tokens"`
	Temperature *float64         `json:"temperature,omitempty"`
	Tools       []anthropicTool  `json:"tools,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
}

type anthropicMsg struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []contentBlock
}

type contentBlock struct {
	Type      string         `json:"type"`
	Text      string         `json:"text,omitempty"`
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
	ToolUseID string         `json:"tool_use_id,omitempty"`
	Content   string         `json:"content,omitempty"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// anthropicResponse is the API response format.
type anthropicResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []contentBlock `json:"content"`
	Model        string         `json:"model"`
	StopReason   string         `json:"stop_reason"`
	StopSequence string         `json:"stop_sequence"`
	Usage        struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// Generate sends a request and returns the complete response.
func (a *AnthropicLLM) Generate(ctx context.Context, messages []vega.Message, tools []vega.ToolSchema) (*vega.LLMResponse, error) {
	start := time.Now()

	// Build request
	req := a.buildRequest(messages, tools, false)

	// Make request
	resp, err := a.doRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	// Parse response
	return a.parseResponse(resp, time.Since(start))
}

// GenerateStream sends a request and returns a channel of streaming events.
func (a *AnthropicLLM) GenerateStream(ctx context.Context, messages []vega.Message, tools []vega.ToolSchema) (<-chan vega.StreamEvent, error) {
	// Build request
	req := a.buildRequest(messages, tools, true)

	// Make streaming request
	eventCh := make(chan vega.StreamEvent, 100)

	go func() {
		defer close(eventCh)

		httpReq, err := a.createHTTPRequest(ctx, req)
		if err != nil {
			eventCh <- vega.StreamEvent{Type: vega.StreamEventError, Error: err}
			return
		}

		httpResp, err := a.httpClient.Do(httpReq)
		if err != nil {
			eventCh <- vega.StreamEvent{Type: vega.StreamEventError, Error: err}
			return
		}
		defer httpResp.Body.Close()

		if httpResp.StatusCode != http.StatusOK {
			body, readErr := io.ReadAll(httpResp.Body)
			if readErr != nil {
				eventCh <- vega.StreamEvent{
					Type:  vega.StreamEventError,
					Error: fmt.Errorf("API error %d (failed to read body: %v)", httpResp.StatusCode, readErr),
				}
				return
			}
			eventCh <- vega.StreamEvent{
				Type:  vega.StreamEventError,
				Error: fmt.Errorf("API error %d: %s", httpResp.StatusCode, string(body)),
			}
			return
		}

		a.parseSSE(httpResp.Body, eventCh)
	}()

	return eventCh, nil
}

func (a *AnthropicLLM) buildRequest(messages []vega.Message, tools []vega.ToolSchema, stream bool) *anthropicRequest {
	req := &anthropicRequest{
		Model:     a.model,
		MaxTokens: 8192,
		Stream:    stream,
	}

	// Extract system message and convert others
	var anthropicMsgs []anthropicMsg
	for _, msg := range messages {
		if msg.Role == vega.RoleSystem {
			req.System = msg.Content
			continue
		}

		anthropicMsgs = append(anthropicMsgs, anthropicMsg{
			Role:    string(msg.Role),
			Content: msg.Content,
		})
	}
	req.Messages = anthropicMsgs

	// Convert tools
	if len(tools) > 0 {
		for _, t := range tools {
			req.Tools = append(req.Tools, anthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: t.InputSchema,
			})
		}
	}

	return req
}

func (a *AnthropicLLM) createHTTPRequest(ctx context.Context, req *anthropicRequest) (*http.Request, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", a.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	return httpReq, nil
}

func (a *AnthropicLLM) doRequest(ctx context.Context, req *anthropicRequest) (*anthropicResponse, error) {
	httpReq, err := a.createHTTPRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	httpResp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer httpResp.Body.Close()

	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error %d: %s", httpResp.StatusCode, string(body))
	}

	var resp anthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	return &resp, nil
}

func (a *AnthropicLLM) parseResponse(resp *anthropicResponse, latency time.Duration) (*vega.LLMResponse, error) {
	result := &vega.LLMResponse{
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		LatencyMs:    latency.Milliseconds(),
	}

	// Calculate cost
	result.CostUSD = vega.CalculateCost(resp.Model, result.InputTokens, result.OutputTokens)

	// Parse stop reason
	switch resp.StopReason {
	case "end_turn":
		result.StopReason = vega.StopReasonEnd
	case "tool_use":
		result.StopReason = vega.StopReasonToolUse
	case "max_tokens":
		result.StopReason = vega.StopReasonLength
	case "stop_sequence":
		result.StopReason = vega.StopReasonStop
	}

	// Parse content blocks
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			result.Content += block.Text
		case "tool_use":
			result.ToolCalls = append(result.ToolCalls, vega.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: block.Input,
			})
		}
	}

	return result, nil
}

func (a *AnthropicLLM) parseSSE(reader io.Reader, eventCh chan<- vega.StreamEvent) {
	scanner := bufio.NewScanner(reader)
	var currentEvent string
	var currentData strings.Builder

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			currentData.WriteString(strings.TrimPrefix(line, "data: "))
			continue
		}

		if line == "" && currentEvent != "" {
			// Process complete event
			a.processSSEEvent(currentEvent, currentData.String(), eventCh)
			currentEvent = ""
			currentData.Reset()
		}
	}
}

func (a *AnthropicLLM) processSSEEvent(eventType, data string, eventCh chan<- vega.StreamEvent) {
	switch eventType {
	case "message_start":
		var msg struct {
			Message struct {
				Usage struct {
					InputTokens int `json:"input_tokens"`
				} `json:"usage"`
			} `json:"message"`
		}
		json.Unmarshal([]byte(data), &msg)
		eventCh <- vega.StreamEvent{
			Type:        vega.StreamEventMessageStart,
			InputTokens: msg.Message.Usage.InputTokens,
		}

	case "content_block_start":
		var block struct {
			ContentBlock struct {
				Type  string `json:"type"`
				ID    string `json:"id"`
				Name  string `json:"name"`
			} `json:"content_block"`
		}
		json.Unmarshal([]byte(data), &block)
		if block.ContentBlock.Type == "tool_use" {
			eventCh <- vega.StreamEvent{
				Type: vega.StreamEventToolStart,
				ToolCall: &vega.ToolCall{
					ID:        block.ContentBlock.ID,
					Name:      block.ContentBlock.Name,
					Arguments: make(map[string]any),
				},
			}
		} else {
			eventCh <- vega.StreamEvent{Type: vega.StreamEventContentStart}
		}

	case "content_block_delta":
		var delta struct {
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		json.Unmarshal([]byte(data), &delta)
		switch delta.Delta.Type {
		case "text_delta":
			eventCh <- vega.StreamEvent{
				Type:  vega.StreamEventContentDelta,
				Delta: delta.Delta.Text,
			}
		case "input_json_delta":
			eventCh <- vega.StreamEvent{
				Type:  vega.StreamEventToolDelta,
				Delta: delta.Delta.PartialJSON,
			}
		}

	case "content_block_stop":
		eventCh <- vega.StreamEvent{Type: vega.StreamEventContentEnd}

	case "message_delta":
		var delta struct {
			Usage struct {
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}
		json.Unmarshal([]byte(data), &delta)
		eventCh <- vega.StreamEvent{
			Type:         vega.StreamEventMessageEnd,
			OutputTokens: delta.Usage.OutputTokens,
		}

	case "message_stop":
		// Final event, no action needed

	case "error":
		var errResp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		json.Unmarshal([]byte(data), &errResp)
		eventCh <- vega.StreamEvent{
			Type:  vega.StreamEventError,
			Error: fmt.Errorf("stream error: %s", errResp.Error.Message),
		}
	}
}
