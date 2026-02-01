package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
)

// StdioTransport implements Transport over subprocess stdin/stdout.
type StdioTransport struct {
	config  ServerConfig
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  io.ReadCloser
	stderr  io.ReadCloser

	// Request tracking
	nextID   int64
	pending  map[int64]chan *JSONRPCResponse
	pendingMu sync.Mutex

	// Notification handling
	notifyHandler func(method string, params json.RawMessage)
	notifyMu      sync.RWMutex

	// Lifecycle
	done     chan struct{}
	closeErr error
	mu       sync.Mutex
}

// NewStdioTransport creates a new stdio transport.
func NewStdioTransport(config ServerConfig) *StdioTransport {
	return &StdioTransport{
		config:  config,
		pending: make(map[int64]chan *JSONRPCResponse),
		done:    make(chan struct{}),
	}
}

// Connect starts the subprocess and establishes communication.
func (t *StdioTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Create command
	t.cmd = exec.CommandContext(ctx, t.config.Command, t.config.Args...)

	// Set environment
	if len(t.config.Env) > 0 {
		t.cmd.Env = os.Environ()
		for k, v := range t.config.Env {
			t.cmd.Env = append(t.cmd.Env, k+"="+v)
		}
	}

	// Get pipes
	var err error
	t.stdin, err = t.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("stdin pipe: %w", err)
	}

	t.stdout, err = t.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}

	t.stderr, err = t.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}

	// Start process
	if err := t.cmd.Start(); err != nil {
		return fmt.Errorf("start process: %w", err)
	}

	// Start reading responses
	go t.readLoop()

	// Start reading stderr for debugging
	go t.readStderr()

	return nil
}

// Send sends a JSON-RPC request and waits for the response.
func (t *StdioTransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := atomic.AddInt64(&t.nextID, 1)

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	// Create response channel
	respCh := make(chan *JSONRPCResponse, 1)

	t.pendingMu.Lock()
	t.pending[id] = respCh
	t.pendingMu.Unlock()

	defer func() {
		t.pendingMu.Lock()
		ch := t.pending[id]
		delete(t.pending, id)
		t.pendingMu.Unlock()
		// Close channel to prevent goroutine leaks if response arrives after cleanup
		if ch != nil {
			// Drain channel if needed
			select {
			case <-ch:
			default:
			}
		}
	}()

	// Send request
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	t.mu.Lock()
	if t.stdin == nil {
		t.mu.Unlock()
		return nil, fmt.Errorf("transport not connected")
	}
	_, err = fmt.Fprintf(t.stdin, "%s\n", data)
	t.mu.Unlock()

	if err != nil {
		return nil, fmt.Errorf("write request: %w", err)
	}

	// Wait for response
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-t.done:
		return nil, fmt.Errorf("transport closed")
	case resp := <-respCh:
		if resp.Error != nil {
			return nil, resp.Error
		}
		return resp.Result, nil
	}
}

// Close shuts down the transport.
func (t *StdioTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Signal shutdown
	select {
	case <-t.done:
		// Already closed
		return t.closeErr
	default:
		close(t.done)
	}

	// Close stdin to signal subprocess
	if t.stdin != nil {
		t.stdin.Close()
	}

	// Wait for process to exit
	if t.cmd != nil && t.cmd.Process != nil {
		t.closeErr = t.cmd.Wait()
	}

	return t.closeErr
}

// OnNotification registers a handler for server notifications.
func (t *StdioTransport) OnNotification(handler func(method string, params json.RawMessage)) {
	t.notifyMu.Lock()
	defer t.notifyMu.Unlock()
	t.notifyHandler = handler
}

// readLoop reads JSON-RPC messages from stdout.
func (t *StdioTransport) readLoop() {
	scanner := bufio.NewScanner(t.stdout)
	// Increase buffer size for large responses
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// Try to parse as response first
		var resp JSONRPCResponse
		if err := json.Unmarshal(line, &resp); err == nil && resp.ID != 0 {
			t.pendingMu.Lock()
			if ch, ok := t.pending[resp.ID]; ok {
				ch <- &resp
			}
			t.pendingMu.Unlock()
			continue
		}

		// Try to parse as notification
		var notif JSONRPCNotification
		if err := json.Unmarshal(line, &notif); err == nil && notif.Method != "" {
			t.notifyMu.RLock()
			handler := t.notifyHandler
			t.notifyMu.RUnlock()

			if handler != nil {
				params, _ := json.Marshal(notif.Params)
				go handler(notif.Method, params)
			}
		}
	}

	// Close all pending requests
	t.pendingMu.Lock()
	for _, ch := range t.pending {
		close(ch)
	}
	t.pending = make(map[int64]chan *JSONRPCResponse)
	t.pendingMu.Unlock()
}

// readStderr reads stderr and logs it for debugging.
func (t *StdioTransport) readStderr() {
	scanner := bufio.NewScanner(t.stderr)
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			slog.Debug("mcp server stderr",
				"server", t.config.Name,
				"line", line,
			)
		}
	}
}
