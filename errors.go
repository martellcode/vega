package vega

import (
	"errors"
	"net/http"
	"strings"
)

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

// APIError represents an error from an API call with status information.
type APIError struct {
	StatusCode int
	Message    string
	Err        error
}

func (e *APIError) Error() string {
	if e.Err != nil {
		return e.Message + ": " + e.Err.Error()
	}
	return e.Message
}

func (e *APIError) Unwrap() error {
	return e.Err
}

// ClassifyError determines the ErrorClass for an error.
// This enables intelligent retry decisions based on error type.
func ClassifyError(err error) ErrorClass {
	if err == nil {
		return ErrClassTemporary
	}

	errStr := strings.ToLower(err.Error())

	// Check for API errors with status codes
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return classifyStatusCode(apiErr.StatusCode)
	}

	// Check for rate limiting indicators
	if strings.Contains(errStr, "rate limit") ||
		strings.Contains(errStr, "rate_limit") ||
		strings.Contains(errStr, "too many requests") ||
		strings.Contains(errStr, "429") {
		return ErrClassRateLimit
	}

	// Check for overloaded/capacity indicators
	if strings.Contains(errStr, "overloaded") ||
		strings.Contains(errStr, "capacity") ||
		strings.Contains(errStr, "503") ||
		strings.Contains(errStr, "service unavailable") {
		return ErrClassOverloaded
	}

	// Check for timeout indicators
	if strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "deadline exceeded") ||
		strings.Contains(errStr, "context canceled") {
		return ErrClassTimeout
	}

	// Check for authentication errors
	if strings.Contains(errStr, "unauthorized") ||
		strings.Contains(errStr, "authentication") ||
		strings.Contains(errStr, "invalid api key") ||
		strings.Contains(errStr, "401") {
		return ErrClassAuthentication
	}

	// Check for invalid request errors
	if strings.Contains(errStr, "invalid") ||
		strings.Contains(errStr, "bad request") ||
		strings.Contains(errStr, "400") ||
		strings.Contains(errStr, "validation") {
		return ErrClassInvalidRequest
	}

	// Check for budget exceeded
	if errors.Is(err, ErrBudgetExceeded) {
		return ErrClassBudgetExceeded
	}

	// Default to temporary (potentially retryable)
	return ErrClassTemporary
}

// classifyStatusCode maps HTTP status codes to ErrorClass.
func classifyStatusCode(code int) ErrorClass {
	switch code {
	case http.StatusTooManyRequests:
		return ErrClassRateLimit
	case http.StatusServiceUnavailable:
		return ErrClassOverloaded
	case http.StatusGatewayTimeout, http.StatusRequestTimeout:
		return ErrClassTimeout
	case http.StatusUnauthorized, http.StatusForbidden:
		return ErrClassAuthentication
	case http.StatusBadRequest:
		return ErrClassInvalidRequest
	default:
		if code >= 500 {
			return ErrClassTemporary
		}
		return ErrClassInvalidRequest
	}
}

// IsRetryable returns true if the error class should typically be retried.
func IsRetryable(class ErrorClass) bool {
	switch class {
	case ErrClassRateLimit, ErrClassOverloaded, ErrClassTimeout, ErrClassTemporary:
		return true
	case ErrClassInvalidRequest, ErrClassAuthentication, ErrClassBudgetExceeded:
		return false
	default:
		return false
	}
}

// ShouldRetry checks if an error should be retried based on the retry policy.
func ShouldRetry(err error, policy *RetryPolicy, attempt int) bool {
	if policy == nil || attempt >= policy.MaxAttempts {
		return false
	}

	class := ClassifyError(err)

	// If RetryOn is specified, only retry those classes
	if len(policy.RetryOn) > 0 {
		for _, retryClass := range policy.RetryOn {
			if retryClass == class {
				return true
			}
		}
		return false
	}

	// Default: retry anything that's retryable
	return IsRetryable(class)
}
