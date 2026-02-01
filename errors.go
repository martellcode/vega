package vega

import "errors"

// Standard errors
var (
	// ErrProcessNotRunning is returned when trying to send to a stopped process
	ErrProcessNotRunning = errors.New("process is not running")

	// ErrNotCompleted is returned when accessing Future result before completion
	ErrNotCompleted = errors.New("operation not completed")

	// ErrMaxIterationsExceeded is returned when tool loop exceeds safety limit
	ErrMaxIterationsExceeded = errors.New("maximum iterations exceeded")

	// ErrBudgetExceeded is returned when cost would exceed budget
	ErrBudgetExceeded = errors.New("budget exceeded")

	// ErrRateLimited is returned when rate limit is hit
	ErrRateLimited = errors.New("rate limited")

	// ErrCircuitOpen is returned when circuit breaker is open
	ErrCircuitOpen = errors.New("circuit breaker is open")

	// ErrToolNotFound is returned when a tool is not registered
	ErrToolNotFound = errors.New("tool not found")

	// ErrSandboxViolation is returned when file access escapes sandbox
	ErrSandboxViolation = errors.New("sandbox violation: path escapes allowed directory")

	// ErrMaxProcessesReached is returned when orchestrator is at capacity
	ErrMaxProcessesReached = errors.New("maximum number of processes reached")

	// ErrProcessNotFound is returned when process ID is not found
	ErrProcessNotFound = errors.New("process not found")

	// ErrAgentNotFound is returned when agent name is not found
	ErrAgentNotFound = errors.New("agent not found")

	// ErrWorkflowNotFound is returned when workflow name is not found
	ErrWorkflowNotFound = errors.New("workflow not found")

	// ErrInvalidInput is returned for invalid workflow inputs
	ErrInvalidInput = errors.New("invalid input")

	// ErrTimeout is returned when an operation times out
	ErrTimeout = errors.New("operation timed out")

	// ErrLinkedProcessDied is returned when a process dies due to a linked process dying
	ErrLinkedProcessDied = errors.New("linked process died")

	// ErrNameTaken is returned when trying to register a name that's already in use
	ErrNameTaken = errors.New("name already registered")

	// ErrGroupNotFound is returned when a process group doesn't exist
	ErrGroupNotFound = errors.New("process group not found")
)

// ProcessError wraps errors with process context.
type ProcessError struct {
	ProcessID string
	AgentName string
	Err       error
}

func (e *ProcessError) Error() string {
	return "process " + e.ProcessID + " (" + e.AgentName + "): " + e.Err.Error()
}

func (e *ProcessError) Unwrap() error {
	return e.Err
}

// ToolError wraps errors with tool context.
type ToolError struct {
	ToolName string
	Err      error
}

func (e *ToolError) Error() string {
	return "tool " + e.ToolName + ": " + e.Err.Error()
}

func (e *ToolError) Unwrap() error {
	return e.Err
}

// ValidationError provides detailed validation failure information.
type ValidationError struct {
	Field   string
	Message string
	Line    int
	Column  int
}

func (e *ValidationError) Error() string {
	if e.Line > 0 {
		return e.Field + " at line " + string(rune(e.Line)) + ": " + e.Message
	}
	return e.Field + ": " + e.Message
}
