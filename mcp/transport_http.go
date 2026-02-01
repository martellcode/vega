package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
)

// HTTPTransport implements Transport over HTTP with optional SSE.
type HTTPTransport struct {
	config  ServerConfig
	client  *http.Client

	// Request tracking
	nextID int64

	// Notification handling
	notifyHandler func(method string, params json.RawMessage)
	notifyMu      sync.RWMutex

	// SSE connection
	sseCancel context.CancelFunc

	mu sync.Mutex
}

// NewHTTPTransport creates a new HTTP transport.
func NewHTTPTransport(config ServerConfig) *HTTPTransport {
	return &HTTPTransport{
		config: config,
		client: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

// Connect establishes the HTTP connection.
func (t *HTTPTransport) Connect(ctx context.Context) error {
	// HTTP is stateless, but we can optionally start SSE for notifications
	if t.config.Transport == TransportSSE {
		go t.startSSE(ctx)
	}
	return nil
}

// Send sends a JSON-RPC request over HTTP.
func (t *HTTPTransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := atomic.AddInt64(&t.nextID, 1)

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", t.config.URL, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range t.config.Headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := t.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf("http status %d (failed to read body: %v)", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("http status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var rpcResp JSONRPCResponse
	if err := json.Unmarshal(body, &rpcResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	if rpcResp.Error != nil {
		return nil, rpcResp.Error
	}

	return rpcResp.Result, nil
}

// Close closes the HTTP transport.
func (t *HTTPTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.sseCancel != nil {
		t.sseCancel()
	}

	return nil
}

// OnNotification registers a handler for server notifications.
func (t *HTTPTransport) OnNotification(handler func(method string, params json.RawMessage)) {
	t.notifyMu.Lock()
	defer t.notifyMu.Unlock()
	t.notifyHandler = handler
}

// startSSE starts listening for Server-Sent Events.
func (t *HTTPTransport) startSSE(ctx context.Context) {
	t.mu.Lock()
	ctx, t.sseCancel = context.WithCancel(ctx)
	t.mu.Unlock()

	// Build SSE URL (typically same base with /events or /sse suffix)
	sseURL := t.config.URL + "/events"

	req, err := http.NewRequestWithContext(ctx, "GET", sseURL, nil)
	if err != nil {
		return
	}

	req.Header.Set("Accept", "text/event-stream")
	for k, v := range t.config.Headers {
		req.Header.Set(k, v)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	// Read SSE events
	// This is a simplified SSE parser
	buf := make([]byte, 4096)
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		n, err := resp.Body.Read(buf)
		if err != nil {
			return
		}

		// Parse SSE event (simplified)
		data := string(buf[:n])
		// In a full implementation, we'd parse the SSE format properly
		_ = data
	}
}
