package vega

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"time"
)

// Process is a running Agent with state and lifecycle.
type Process struct {
	// ID is the unique identifier for this process
	ID string

	// Agent is the agent definition this process is running
	Agent *Agent

	// Task describes what this process is working on
	Task string

	// WorkDir is the isolated workspace directory
	WorkDir string

	// Project is the container project name for isolated execution
	Project string

	// StartedAt is when the process was spawned
	StartedAt time.Time

	// Supervision configures fault tolerance
	Supervision *Supervision

	// status is the current process state
	status Status

	// metrics tracks usage
	metrics ProcessMetrics

	// context for cancellation
	ctx    context.Context
	cancel context.CancelFunc

	// messages is the conversation history
	messages []Message

	// iteration count
	iteration int

	// llm is the backend to use
	llm LLM

	// orchestrator reference for child spawning
	orchestrator *Orchestrator

	// mutex for thread safety
	mu sync.RWMutex

	// result channel for async operations
	resultCh chan *SendResult

	// finalResult stores the result when process completes
	finalResult string

	// Process linking (Erlang-style)
	// links are bidirectional - if linked process dies, we die too (unless trapExit)
	links map[string]*Process
	// monitors are unidirectional - we get notified when monitored process dies
	monitors map[string]*monitorEntry
	// monitoredBy tracks who is monitoring us (for cleanup)
	monitoredBy map[string]*monitorEntry
	// trapExit when true, converts exit signals to messages instead of killing
	trapExit bool
	// exitSignals receives exit notifications when trapExit is true
	exitSignals chan ExitSignal
	// linkMu protects link/monitor maps
	linkMu sync.RWMutex
	// nextMonitorID for generating unique monitor references
	nextMonitorID uint64

	// Named process support
	name string

	// Automatic restart support
	restartPolicy ChildRestart
	spawnOpts     []SpawnOption

	// Process group membership
	groups map[string]*ProcessGroup
}

// Status represents the process lifecycle state.
type Status string

const (
	StatusPending   Status = "pending"
	StatusRunning   Status = "running"
	StatusCompleted Status = "completed"
	StatusFailed    Status = "failed"
	StatusTimeout   Status = "timeout"
)

// ExitReason describes why a process exited.
type ExitReason string

const (
	// ExitNormal means the process completed successfully
	ExitNormal ExitReason = "normal"
	// ExitError means the process failed with an error
	ExitError ExitReason = "error"
	// ExitKilled means the process was explicitly killed
	ExitKilled ExitReason = "killed"
	// ExitLinked means the process died because a linked process died
	ExitLinked ExitReason = "linked"
)

// ExitSignal is sent to linked/monitoring processes when a process exits.
// When trapExit is true, these are delivered via the ExitSignals channel.
// When trapExit is false, linked process deaths cause this process to die.
type ExitSignal struct {
	// ProcessID is the ID of the process that exited
	ProcessID string
	// AgentName is the name of the agent that was running
	AgentName string
	// Reason explains why the process exited
	Reason ExitReason
	// Error is set if Reason is ExitError
	Error error
	// Result is set if Reason is ExitNormal
	Result string
	// Timestamp is when the exit occurred
	Timestamp time.Time
}

// MonitorRef is a reference to an active monitor, used for demonitoring.
type MonitorRef struct {
	id        uint64
	processID string
}

// monitorEntry tracks a monitoring relationship.
type monitorEntry struct {
	ref     MonitorRef
	process *Process
}

// ProcessMetrics tracks process usage.
type ProcessMetrics struct {
	Iterations   int
	InputTokens  int
	OutputTokens int
	CostUSD      float64
	StartedAt    time.Time
	CompletedAt  time.Time
	LastActiveAt time.Time
	ToolCalls    int
	Errors       int
}

// SendResult is the result of a Send operation.
type SendResult struct {
	Response string
	Error    error
	Metrics  CallMetrics
}

// CallMetrics tracks a single LLM call.
type CallMetrics struct {
	InputTokens  int
	OutputTokens int
	CostUSD      float64
	LatencyMs    int64
	ToolCalls    []string
	Retries      int
}

// Status returns the current process status.
func (p *Process) Status() Status {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.status
}

// Metrics returns the current process metrics.
func (p *Process) Metrics() ProcessMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.metrics
}

// Name returns the registered name of the process, or empty string if not named.
func (p *Process) Name() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.name
}

// Groups returns the names of all groups this process belongs to.
func (p *Process) Groups() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	names := make([]string, 0, len(p.groups))
	for name := range p.groups {
		names = append(names, name)
	}
	return names
}

// Send sends a message and waits for a response.
func (p *Process) Send(ctx context.Context, message string) (string, error) {
	p.mu.Lock()
	if p.status != StatusRunning && p.status != StatusPending {
		p.mu.Unlock()
		return "", ErrProcessNotRunning
	}
	p.status = StatusRunning
	p.iteration++
	p.metrics.LastActiveAt = time.Now()
	p.mu.Unlock()

	// Add user message to context
	p.addMessage(Message{Role: RoleUser, Content: message})

	// Execute the LLM call loop (may involve tool calls)
	response, callMetrics, err := p.executeLLMLoop(ctx, message)
	if err != nil {
		p.mu.Lock()
		p.metrics.Errors++
		p.mu.Unlock()
		return "", err
	}

	// Update metrics
	p.mu.Lock()
	p.metrics.InputTokens += callMetrics.InputTokens
	p.metrics.OutputTokens += callMetrics.OutputTokens
	p.metrics.CostUSD += callMetrics.CostUSD
	p.metrics.ToolCalls += len(callMetrics.ToolCalls)
	p.mu.Unlock()

	// Add assistant response to context
	p.addMessage(Message{Role: RoleAssistant, Content: response})

	return response, nil
}

// SendAsync sends a message and returns a Future.
func (p *Process) SendAsync(message string) *Future {
	f := &Future{
		done:   make(chan struct{}),
		cancel: make(chan struct{}),
	}

	go func() {
		ctx, cancel := context.WithCancel(context.Background())

		// Handle cancellation
		go func() {
			select {
			case <-f.cancel:
				cancel()
			case <-f.done:
			}
		}()

		result, err := p.Send(ctx, message)
		f.mu.Lock()
		f.result = result
		f.err = err
		f.completed = true
		f.mu.Unlock()
		close(f.done)
	}()

	return f
}

// SendStream sends a message and returns a streaming response.
func (p *Process) SendStream(ctx context.Context, message string) (*Stream, error) {
	p.mu.Lock()
	if p.status != StatusRunning && p.status != StatusPending {
		p.mu.Unlock()
		return nil, ErrProcessNotRunning
	}
	p.status = StatusRunning
	p.iteration++
	p.metrics.LastActiveAt = time.Now()
	p.mu.Unlock()

	// Add user message to context
	p.addMessage(Message{Role: RoleUser, Content: message})

	// Create stream
	stream := &Stream{
		chunks: make(chan string, 100),
		done:   make(chan struct{}),
	}

	// Execute streaming in goroutine
	go func() {
		defer close(stream.chunks)
		defer close(stream.done)

		response, err := p.executeLLMStream(ctx, message, stream.chunks)
		stream.mu.Lock()
		stream.response = response
		stream.err = err
		stream.mu.Unlock()

		// Add assistant response to context
		if err == nil {
			p.addMessage(Message{Role: RoleAssistant, Content: response})
		}
	}()

	return stream, nil
}

// Stop terminates the process.
// This is equivalent to killing the process - linked processes will be notified.
func (p *Process) Stop() {
	p.mu.Lock()
	if p.status == StatusCompleted || p.status == StatusFailed {
		p.mu.Unlock()
		return // Already dead
	}

	if p.cancel != nil {
		p.cancel()
	}
	p.status = StatusCompleted
	p.metrics.CompletedAt = time.Now()
	agentName := ""
	if p.Agent != nil {
		agentName = p.Agent.Name
	}
	p.mu.Unlock()

	// Propagate exit to linked/monitoring processes
	signal := ExitSignal{
		ProcessID: p.ID,
		AgentName: agentName,
		Reason:    ExitKilled,
		Timestamp: time.Now(),
	}
	p.propagateExit(signal)

	// Notify orchestrator (for name unregistration)
	if p.orchestrator != nil {
		p.orchestrator.emitComplete(p, "")
	}
}

// Complete marks the process as successfully completed with a result.
// This triggers OnProcessComplete callbacks and notifies linked/monitoring processes.
// Normal completion does NOT cause linked processes to die.
func (p *Process) Complete(result string) {
	p.mu.Lock()
	if p.status == StatusCompleted || p.status == StatusFailed {
		p.mu.Unlock()
		return // Already finished
	}

	if p.cancel != nil {
		p.cancel()
	}
	p.status = StatusCompleted
	p.finalResult = result
	p.metrics.CompletedAt = time.Now()
	agentName := ""
	if p.Agent != nil {
		agentName = p.Agent.Name
	}
	p.mu.Unlock()

	// Propagate exit to linked/monitoring processes (normal exit)
	signal := ExitSignal{
		ProcessID: p.ID,
		AgentName: agentName,
		Reason:    ExitNormal,
		Result:    result,
		Timestamp: time.Now(),
	}
	p.propagateExit(signal)

	// Notify orchestrator
	if p.orchestrator != nil {
		p.orchestrator.emitComplete(p, result)
	}
}

// Fail marks the process as failed with an error.
// This triggers OnProcessFailed callbacks and notifies linked/monitoring processes.
// Failed processes cause linked processes to die too (unless they trap exits).
func (p *Process) Fail(err error) {
	p.mu.Lock()
	if p.status == StatusCompleted || p.status == StatusFailed {
		p.mu.Unlock()
		return // Already finished
	}

	if p.cancel != nil {
		p.cancel()
	}
	p.status = StatusFailed
	p.metrics.CompletedAt = time.Now()
	p.metrics.Errors++
	agentName := ""
	if p.Agent != nil {
		agentName = p.Agent.Name
	}
	p.mu.Unlock()

	// Propagate exit to linked/monitoring processes (error exit)
	signal := ExitSignal{
		ProcessID: p.ID,
		AgentName: agentName,
		Reason:    ExitError,
		Error:     err,
		Timestamp: time.Now(),
	}
	p.propagateExit(signal)

	// Notify orchestrator
	if p.orchestrator != nil {
		p.orchestrator.emitFailed(p, err)
	}
}

// Result returns the final result if the process completed.
func (p *Process) Result() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.finalResult
}

// addMessage adds a message to the conversation history.
func (p *Process) addMessage(msg Message) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.Agent.Context != nil {
		p.Agent.Context.Add(msg)
	}
	p.messages = append(p.messages, msg)
}

// executeLLMLoop runs the LLM call loop, handling tool calls.
func (p *Process) executeLLMLoop(ctx context.Context, message string) (string, CallMetrics, error) {
	metrics := CallMetrics{}

	// Build messages for LLM
	messages := p.buildMessages()

	// Get tools schema if agent has tools
	var toolSchemas []ToolSchema
	if p.Agent.Tools != nil {
		toolSchemas = p.Agent.Tools.Schema()
	}

	// Main loop - keep calling LLM until we get a final response (no tool calls)
	maxIterations := 50 // Safety limit
	for i := 0; i < maxIterations; i++ {
		select {
		case <-ctx.Done():
			return "", metrics, ctx.Err()
		default:
		}

		// Call LLM
		resp, err := p.llm.Generate(ctx, messages, toolSchemas)
		if err != nil {
			return "", metrics, err
		}

		// Update metrics
		metrics.InputTokens += resp.InputTokens
		metrics.OutputTokens += resp.OutputTokens
		metrics.CostUSD += resp.CostUSD
		metrics.LatencyMs += resp.LatencyMs

		// If no tool calls, we're done
		if len(resp.ToolCalls) == 0 {
			return resp.Content, metrics, nil
		}

		// Execute tool calls
		messages = append(messages, Message{Role: RoleAssistant, Content: resp.Content})

		for _, tc := range resp.ToolCalls {
			metrics.ToolCalls = append(metrics.ToolCalls, tc.Name)

			result, err := p.Agent.Tools.Execute(ctx, tc.Name, tc.Arguments)
			if err != nil {
				result = "Error: " + err.Error()
			}

			// Add tool result as user message (this is how Anthropic expects it)
			messages = append(messages, Message{
				Role:    RoleUser,
				Content: formatToolResult(tc.ID, tc.Name, result),
			})
		}
	}

	return "", metrics, ErrMaxIterationsExceeded
}

// executeLLMStream runs streaming LLM call with tool execution loop.
func (p *Process) executeLLMStream(ctx context.Context, message string, chunks chan<- string) (string, error) {
	messages := p.buildMessages()

	var toolSchemas []ToolSchema
	if p.Agent.Tools != nil {
		toolSchemas = p.Agent.Tools.Schema()
	}

	var fullResponse string
	maxIterations := 50 // Safety limit

	for i := 0; i < maxIterations; i++ {
		select {
		case <-ctx.Done():
			return fullResponse, ctx.Err()
		default:
		}

		eventCh, err := p.llm.GenerateStream(ctx, messages, toolSchemas)
		if err != nil {
			return fullResponse, err
		}

		// Collect response and tool calls from this iteration
		var iterResponse string
		var toolCalls []ToolCall
		var currentToolCall *ToolCall
		var currentToolJSON string

		for event := range eventCh {
			if event.Error != nil {
				return fullResponse, event.Error
			}

			switch event.Type {
			case StreamEventContentDelta:
				if event.Delta != "" {
					chunks <- event.Delta
					iterResponse += event.Delta
					fullResponse += event.Delta
				}
			case StreamEventToolStart:
				if event.ToolCall != nil {
					currentToolCall = &ToolCall{
						ID:        event.ToolCall.ID,
						Name:      event.ToolCall.Name,
						Arguments: make(map[string]any),
					}
					currentToolJSON = ""
				}
			case StreamEventToolDelta:
				if currentToolCall != nil {
					currentToolJSON += event.Delta
				}
			case StreamEventContentEnd:
				// If we were building a tool call, finalize it
				if currentToolCall != nil {
					if currentToolJSON != "" {
						json.Unmarshal([]byte(currentToolJSON), &currentToolCall.Arguments)
					}
					toolCalls = append(toolCalls, *currentToolCall)
					currentToolCall = nil
					currentToolJSON = ""
				}
			}
		}

		// If no tool calls, we're done
		if len(toolCalls) == 0 {
			return fullResponse, nil
		}

		// Add assistant message with the response (include tool call info if no text)
		assistantContent := iterResponse
		if assistantContent == "" {
			// Build a representation of the tool calls for the message
			var toolParts []string
			for _, tc := range toolCalls {
				toolParts = append(toolParts, formatToolCall(tc.ID, tc.Name, tc.Arguments))
			}
			assistantContent = strings.Join(toolParts, "\n")
		}
		messages = append(messages, Message{Role: RoleAssistant, Content: assistantContent})

		// Execute tool calls and add results
		for _, tc := range toolCalls {
			p.mu.Lock()
			p.metrics.ToolCalls++
			p.mu.Unlock()

			result, err := p.Agent.Tools.Execute(ctx, tc.Name, tc.Arguments)
			if err != nil {
				result = "Error: " + err.Error()
			}

			// Add tool result as user message
			messages = append(messages, Message{
				Role:    RoleUser,
				Content: formatToolResult(tc.ID, tc.Name, result),
			})
		}
	}

	return fullResponse, ErrMaxIterationsExceeded
}

// buildMessages builds the message list for LLM call.
func (p *Process) buildMessages() []Message {
	var messages []Message

	// Set skill context if using SkillsPrompt
	if sp, ok := p.Agent.System.(*SkillsPrompt); ok {
		p.mu.RLock()
		if len(p.messages) > 0 {
			// Find the last user message
			for i := len(p.messages) - 1; i >= 0; i-- {
				if p.messages[i].Role == RoleUser {
					sp.SetContext(p.messages[i].Content)
					break
				}
			}
		}
		p.mu.RUnlock()
	}

	// Add system prompt
	if p.Agent.System != nil {
		messages = append(messages, Message{
			Role:    RoleSystem,
			Content: p.Agent.System.Prompt(),
		})
	}

	// Add conversation history
	if p.Agent.Context != nil {
		maxTokens := 100000 // Default, could be configurable
		if p.Agent.MaxTokens > 0 {
			maxTokens = p.Agent.MaxTokens
		}
		messages = append(messages, p.Agent.Context.Messages(maxTokens)...)
	} else {
		p.mu.RLock()
		messages = append(messages, p.messages...)
		p.mu.RUnlock()
	}

	return messages
}

// formatToolResult formats a tool result for the LLM.
func formatToolResult(id, name, result string) string {
	return "<tool_result tool_use_id=\"" + id + "\" name=\"" + name + "\">\n" + result + "\n</tool_result>"
}

// formatToolCall formats a tool call for the assistant message.
func formatToolCall(id, name string, args map[string]any) string {
	argsJSON, _ := json.Marshal(args)
	return "<tool_use id=\"" + id + "\" name=\"" + name + "\">\n" + string(argsJSON) + "\n</tool_use>"
}
// Future represents an asynchronous operation result.
type Future struct {
	result    string
	err       error
	completed bool
	done      chan struct{}
	cancel    chan struct{}
	mu        sync.RWMutex
}

// Await waits for the future to complete and returns the result.
func (f *Future) Await(ctx context.Context) (string, error) {
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-f.done:
		f.mu.RLock()
		defer f.mu.RUnlock()
		return f.result, f.err
	}
}

// Done returns true if the future has completed.
func (f *Future) Done() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.completed
}

// Result returns the result if completed, or error if not.
func (f *Future) Result() (string, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if !f.completed {
		return "", ErrNotCompleted
	}
	return f.result, f.err
}

// Cancel cancels the future.
func (f *Future) Cancel() {
	select {
	case f.cancel <- struct{}{}:
	default:
	}
}

// Stream represents a streaming response.
type Stream struct {
	chunks   chan string
	response string
	err      error
	done     chan struct{}
	mu       sync.RWMutex
}

// Chunks returns the channel of response chunks.
func (s *Stream) Chunks() <-chan string {
	return s.chunks
}

// Response returns the complete response after streaming is done.
func (s *Stream) Response() string {
	<-s.done
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.response
}

// Err returns any error that occurred during streaming.
func (s *Stream) Err() error {
	<-s.done
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.err
}

// --- Process Linking (Erlang-style) ---

// Link creates a bidirectional link between this process and another.
// If either process dies, the other will also die (unless trapExit is set).
// Linking is idempotent - linking to an already-linked process is a no-op.
func (p *Process) Link(other *Process) {
	if p == other || other == nil {
		return
	}

	// Lock both processes in consistent order to avoid deadlock
	first, second := p, other
	if p.ID > other.ID {
		first, second = other, p
	}

	first.linkMu.Lock()
	second.linkMu.Lock()

	// Initialize maps if needed
	if first.links == nil {
		first.links = make(map[string]*Process)
	}
	if second.links == nil {
		second.links = make(map[string]*Process)
	}

	// Create bidirectional link
	first.links[second.ID] = second
	second.links[first.ID] = first

	second.linkMu.Unlock()
	first.linkMu.Unlock()
}

// Unlink removes the bidirectional link between this process and another.
// Unlinking is idempotent - unlinking from a non-linked process is a no-op.
func (p *Process) Unlink(other *Process) {
	if p == other || other == nil {
		return
	}

	// Lock both processes in consistent order to avoid deadlock
	first, second := p, other
	if p.ID > other.ID {
		first, second = other, p
	}

	first.linkMu.Lock()
	second.linkMu.Lock()

	delete(first.links, second.ID)
	delete(second.links, first.ID)

	second.linkMu.Unlock()
	first.linkMu.Unlock()
}

// Monitor starts monitoring another process.
// When the monitored process exits, this process receives an ExitSignal
// on its ExitSignals channel (does not cause death, unlike Link).
// Returns a MonitorRef that can be used to stop monitoring.
func (p *Process) Monitor(other *Process) MonitorRef {
	if p == other || other == nil {
		return MonitorRef{}
	}

	p.linkMu.Lock()
	defer p.linkMu.Unlock()

	other.linkMu.Lock()
	defer other.linkMu.Unlock()

	// Initialize maps if needed
	if p.monitors == nil {
		p.monitors = make(map[string]*monitorEntry)
	}
	if other.monitoredBy == nil {
		other.monitoredBy = make(map[string]*monitorEntry)
	}
	if p.exitSignals == nil {
		p.exitSignals = make(chan ExitSignal, 16)
	}

	// Generate unique monitor ID
	p.nextMonitorID++
	ref := MonitorRef{
		id:        p.nextMonitorID,
		processID: other.ID,
	}

	entry := &monitorEntry{
		ref:     ref,
		process: p,
	}

	p.monitors[other.ID] = entry
	other.monitoredBy[p.ID] = entry

	return ref
}

// Demonitor stops monitoring a process.
// The MonitorRef must be one returned by a previous Monitor call.
func (p *Process) Demonitor(ref MonitorRef) {
	if ref.processID == "" {
		return
	}

	p.linkMu.Lock()
	entry, ok := p.monitors[ref.processID]
	if !ok || entry.ref.id != ref.id {
		p.linkMu.Unlock()
		return
	}
	delete(p.monitors, ref.processID)
	p.linkMu.Unlock()

	// Find the other process and remove ourselves from monitoredBy
	if p.orchestrator != nil {
		if other := p.orchestrator.Get(ref.processID); other != nil {
			other.linkMu.Lock()
			delete(other.monitoredBy, p.ID)
			other.linkMu.Unlock()
		}
	}
}

// SetTrapExit enables or disables exit trapping.
// When trapExit is true, linked process deaths deliver ExitSignals
// instead of killing this process. This is how supervisors survive
// their children dying.
func (p *Process) SetTrapExit(trap bool) {
	p.linkMu.Lock()
	defer p.linkMu.Unlock()

	p.trapExit = trap
	if trap && p.exitSignals == nil {
		p.exitSignals = make(chan ExitSignal, 16)
	}
}

// TrapExit returns whether exit trapping is enabled.
func (p *Process) TrapExit() bool {
	p.linkMu.RLock()
	defer p.linkMu.RUnlock()
	return p.trapExit
}

// ExitSignals returns the channel for receiving exit signals.
// Only receives signals when trapExit is true, or for monitored processes.
// Returns nil if no exit signal channel has been created.
func (p *Process) ExitSignals() <-chan ExitSignal {
	p.linkMu.RLock()
	defer p.linkMu.RUnlock()
	return p.exitSignals
}

// Links returns the IDs of all linked processes.
func (p *Process) Links() []string {
	p.linkMu.RLock()
	defer p.linkMu.RUnlock()

	ids := make([]string, 0, len(p.links))
	for id := range p.links {
		ids = append(ids, id)
	}
	return ids
}

// propagateExit notifies linked and monitoring processes of this process's death.
func (p *Process) propagateExit(signal ExitSignal) {
	p.linkMu.Lock()

	// Collect linked processes
	linkedProcs := make([]*Process, 0, len(p.links))
	for _, linked := range p.links {
		linkedProcs = append(linkedProcs, linked)
	}

	// Collect monitoring processes
	monitoringProcs := make([]*Process, 0, len(p.monitoredBy))
	for _, entry := range p.monitoredBy {
		monitoringProcs = append(monitoringProcs, entry.process)
	}

	// Clear our links and monitors (we're dead)
	p.links = nil
	p.monitoredBy = nil

	p.linkMu.Unlock()

	// Notify linked processes
	for _, linked := range linkedProcs {
		linked.handleLinkedExit(p, signal)
	}

	// Notify monitoring processes
	for _, monitoring := range monitoringProcs {
		monitoring.handleMonitoredExit(p, signal)
	}
}

// handleLinkedExit is called when a linked process dies.
func (p *Process) handleLinkedExit(dead *Process, signal ExitSignal) {
	// Remove the dead process from our links
	p.linkMu.Lock()
	delete(p.links, dead.ID)
	trapExit := p.trapExit
	exitCh := p.exitSignals
	p.linkMu.Unlock()

	if trapExit && exitCh != nil {
		// Trapping exits - deliver as signal instead of dying
		select {
		case exitCh <- signal:
		default:
			// Channel full, signal dropped
		}
		return
	}

	// Not trapping exits - we die too (unless it was a normal exit)
	if signal.Reason == ExitNormal {
		return // Normal exits don't propagate death
	}

	// Cascade the death
	p.mu.Lock()
	if p.status == StatusCompleted || p.status == StatusFailed {
		p.mu.Unlock()
		return // Already dead
	}
	p.status = StatusFailed
	p.metrics.CompletedAt = time.Now()
	p.mu.Unlock()

	// Propagate with ExitLinked reason
	cascadeSignal := ExitSignal{
		ProcessID: p.ID,
		AgentName: p.Agent.Name,
		Reason:    ExitLinked,
		Error:     &LinkedProcessError{LinkedID: dead.ID, OriginalError: signal.Error},
		Timestamp: time.Now(),
	}
	p.propagateExit(cascadeSignal)

	// Notify orchestrator
	if p.orchestrator != nil {
		p.orchestrator.emitFailed(p, cascadeSignal.Error)
	}
}

// handleMonitoredExit is called when a monitored process dies.
func (p *Process) handleMonitoredExit(dead *Process, signal ExitSignal) {
	// Remove the dead process from our monitors
	p.linkMu.Lock()
	delete(p.monitors, dead.ID)
	exitCh := p.exitSignals
	p.linkMu.Unlock()

	// Monitors always deliver signals (never cause death)
	if exitCh != nil {
		select {
		case exitCh <- signal:
		default:
			// Channel full, signal dropped
		}
	}
}

// LinkedProcessError is the error set when a process dies due to a linked process dying.
type LinkedProcessError struct {
	LinkedID      string
	OriginalError error
}

func (e *LinkedProcessError) Error() string {
	if e.OriginalError != nil {
		return "linked process " + e.LinkedID + " died: " + e.OriginalError.Error()
	}
	return "linked process " + e.LinkedID + " died"
}

func (e *LinkedProcessError) Unwrap() error {
	return e.OriginalError
}
