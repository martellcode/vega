package vega

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/martellcode/vega/container"
)

// Orchestrator manages multiple processes.
type Orchestrator struct {
	processes map[string]*Process
	mu        sync.RWMutex

	// Named process registry
	names   map[string]*Process
	namesMu sync.RWMutex

	// Agent registry for respawning
	agents   map[string]Agent
	agentsMu sync.RWMutex

	// Process groups for multi-agent collaboration
	groups   map[string]*ProcessGroup
	groupsMu sync.RWMutex

	// Configuration
	maxProcesses  int
	defaultLLM    LLM
	persistence   Persistence
	healthMonitor *HealthMonitor
	recovery      bool

	// Rate limiting
	rateLimits map[string]*rateLimiter

	// Container management
	containerManager  *container.Manager
	containerRegistry *container.ProjectRegistry

	// Lifecycle callbacks
	onComplete []func(*Process, string)
	onFailed   []func(*Process, error)
	onStarted  []func(*Process)
	callbackMu sync.RWMutex

	// Event callbacks (for distributed workers)
	callbackConfig *CallbackConfig
	eventPoller    *EventPoller

	// Shutdown coordination
	ctx    context.Context
	cancel context.CancelFunc
}

// ProcessEvent represents a process lifecycle event.
type ProcessEvent struct {
	Type      ProcessEventType
	Process   *Process
	Result    string // For complete events
	Error     error  // For failed events
	Timestamp time.Time
}

// ProcessEventType is the type of lifecycle event.
type ProcessEventType int

const (
	ProcessStarted ProcessEventType = iota
	ProcessCompleted
	ProcessFailed
)

// OrchestratorOption configures an Orchestrator.
type OrchestratorOption func(*Orchestrator)

// NewOrchestrator creates a new Orchestrator.
func NewOrchestrator(opts ...OrchestratorOption) *Orchestrator {
	ctx, cancel := context.WithCancel(context.Background())

	o := &Orchestrator{
		processes:    make(map[string]*Process),
		names:        make(map[string]*Process),
		agents:       make(map[string]Agent),
		groups:       make(map[string]*ProcessGroup),
		maxProcesses: 100,
		rateLimits:   make(map[string]*rateLimiter),
		ctx:          ctx,
		cancel:       cancel,
	}

	for _, opt := range opts {
		opt(o)
	}

	// Start health monitoring if configured
	if o.healthMonitor != nil {
		o.healthMonitor.Start(o.List)
	}

	// Recover processes if enabled
	if o.recovery && o.persistence != nil {
		o.recoverProcesses()
	}

	return o
}

// WithMaxProcesses sets the maximum number of concurrent processes.
func WithMaxProcesses(n int) OrchestratorOption {
	return func(o *Orchestrator) {
		o.maxProcesses = n
	}
}

// WithLLM sets the default LLM backend.
func WithLLM(llm LLM) OrchestratorOption {
	return func(o *Orchestrator) {
		o.defaultLLM = llm
	}
}

// WithPersistence enables process state persistence.
func WithPersistence(p Persistence) OrchestratorOption {
	return func(o *Orchestrator) {
		o.persistence = p
	}
}

// WithRecovery enables process recovery on startup.
func WithRecovery(enabled bool) OrchestratorOption {
	return func(o *Orchestrator) {
		o.recovery = enabled
	}
}

// WithHealthCheck enables health monitoring.
func WithHealthCheck(config HealthConfig) OrchestratorOption {
	return func(o *Orchestrator) {
		o.healthMonitor = NewHealthMonitor(config)
	}
}

// WithRateLimits configures per-model rate limiting.
func WithRateLimits(limits map[string]RateLimitConfig) OrchestratorOption {
	return func(o *Orchestrator) {
		for model, config := range limits {
			o.rateLimits[model] = newRateLimiter(config)
		}
	}
}

// WithContainerManager enables container-based project isolation.
// If baseDir is provided, a ProjectRegistry will also be created.
func WithContainerManager(cm *container.Manager, baseDir string) OrchestratorOption {
	return func(o *Orchestrator) {
		o.containerManager = cm
		if baseDir != "" && cm != nil {
			registry, err := container.NewProjectRegistry(baseDir, cm)
			if err == nil {
				o.containerRegistry = registry
			}
		}
	}
}

// RateLimitConfig configures rate limiting for a model.
type RateLimitConfig struct {
	RequestsPerMinute int
	TokensPerMinute   int
	Strategy          RateLimitStrategy
}

// RateLimitStrategy determines rate limit behavior.
type RateLimitStrategy int

const (
	RateLimitQueue RateLimitStrategy = iota
	RateLimitReject
	RateLimitBackpressure
)

// SpawnOption configures a spawned process.
type SpawnOption func(*Process)

// WithTask sets the task description.
func WithTask(task string) SpawnOption {
	return func(p *Process) {
		p.Task = task
	}
}

// WithWorkDir sets the working directory.
func WithWorkDir(dir string) SpawnOption {
	return func(p *Process) {
		p.WorkDir = dir
	}
}

// WithSupervision sets the supervision configuration.
func WithSupervision(s Supervision) SpawnOption {
	return func(p *Process) {
		p.Supervision = &s
	}
}

// WithTimeout sets a timeout for the process.
func WithTimeout(d time.Duration) SpawnOption {
	return func(p *Process) {
		ctx, cancel := context.WithTimeout(p.ctx, d)
		p.ctx = ctx
		oldCancel := p.cancel
		p.cancel = func() {
			cancel()
			if oldCancel != nil {
				oldCancel()
			}
		}
	}
}

// WithMaxIterations sets the maximum iteration count.
func WithMaxIterations(n int) SpawnOption {
	return func(p *Process) {
		// Store in process for checking
		// This is checked in the LLM loop
	}
}

// WithProcessContext sets a parent context.
func WithProcessContext(ctx context.Context) SpawnOption {
	return func(p *Process) {
		p.ctx, p.cancel = context.WithCancel(ctx)
	}
}

// WithProject sets the container project for isolated execution.
func WithProject(name string) SpawnOption {
	return func(p *Process) {
		p.Project = name
	}
}

// Spawn creates and starts a new process from an agent.
func (o *Orchestrator) Spawn(agent Agent, opts ...SpawnOption) (*Process, error) {
	// Validate agent
	if agent.Name == "" {
		return nil, &ProcessError{Err: errors.New("agent name is required")}
	}

	o.mu.Lock()

	// Check capacity
	if len(o.processes) >= o.maxProcesses {
		o.mu.Unlock()
		return nil, ErrMaxProcessesReached
	}

	// Create process
	ctx, cancel := context.WithCancel(o.ctx)
	p := &Process{
		ID:           uuid.New().String()[:8],
		Agent:        &agent,
		status:       StatusPending,
		StartedAt:    time.Now(),
		ctx:          ctx,
		cancel:       cancel,
		orchestrator: o,
		messages:     make([]Message, 0),
		metrics: ProcessMetrics{
			StartedAt: time.Now(),
		},
	}

	// Apply options
	for _, opt := range opts {
		opt(p)
	}

	// Set LLM backend
	if agent.LLM != nil {
		p.llm = agent.LLM
	} else if o.defaultLLM != nil {
		p.llm = o.defaultLLM
	} else {
		o.mu.Unlock()
		return nil, &ProcessError{ProcessID: p.ID, AgentName: agent.Name, Err: ErrProcessNotRunning}
	}

	// Register process
	o.processes[p.ID] = p
	o.mu.Unlock()

	// Persist state
	o.persistState()

	// Mark as running
	p.mu.Lock()
	p.status = StatusRunning
	p.mu.Unlock()

	slog.Info("process spawned",
		"process_id", p.ID,
		"agent", agent.Name,
		"task", p.Task,
	)

	// Emit started event
	o.emitStarted(p)

	return p, nil
}

// Get returns a process by ID.
func (o *Orchestrator) Get(id string) *Process {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.processes[id]
}

// List returns all processes.
func (o *Orchestrator) List() []*Process {
	o.mu.RLock()
	defer o.mu.RUnlock()

	procs := make([]*Process, 0, len(o.processes))
	for _, p := range o.processes {
		procs = append(procs, p)
	}
	return procs
}

// Kill terminates a process.
func (o *Orchestrator) Kill(id string) error {
	o.mu.Lock()
	p, ok := o.processes[id]
	if !ok {
		o.mu.Unlock()
		return ErrProcessNotFound
	}
	o.mu.Unlock()

	p.Stop()

	o.mu.Lock()
	delete(o.processes, id)
	o.mu.Unlock()

	o.persistState()
	return nil
}

// Shutdown gracefully shuts down all processes.
func (o *Orchestrator) Shutdown(ctx context.Context) error {
	// Stop health monitor
	if o.healthMonitor != nil {
		o.healthMonitor.Stop()
	}

	// Stop event poller
	if o.eventPoller != nil {
		o.eventPoller.Stop()
	}

	// Close container manager
	if o.containerManager != nil {
		o.containerManager.Close()
	}

	// Cancel all processes
	o.cancel()

	// Wait for processes to stop or context to expire
	done := make(chan struct{})
	go func() {
		o.mu.RLock()
		for _, p := range o.processes {
			p.Stop()
		}
		o.mu.RUnlock()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// GetContainerManager returns the container manager, if configured.
func (o *Orchestrator) GetContainerManager() *container.Manager {
	return o.containerManager
}

// GetProjectRegistry returns the project registry, if configured.
func (o *Orchestrator) GetProjectRegistry() *container.ProjectRegistry {
	return o.containerRegistry
}

// OnHealthAlert registers a callback for health alerts.
func (o *Orchestrator) OnHealthAlert(fn func(Alert)) {
	if o.healthMonitor == nil {
		return
	}

	go func() {
		for alert := range o.healthMonitor.Alerts() {
			fn(alert)
		}
	}()
}

// OnProcessComplete registers a callback for when a process completes successfully.
// The callback receives the process and its final result.
func (o *Orchestrator) OnProcessComplete(fn func(*Process, string)) {
	o.callbackMu.Lock()
	defer o.callbackMu.Unlock()
	o.onComplete = append(o.onComplete, fn)
}

// OnProcessFailed registers a callback for when a process fails.
// The callback receives the process and the error.
func (o *Orchestrator) OnProcessFailed(fn func(*Process, error)) {
	o.callbackMu.Lock()
	defer o.callbackMu.Unlock()
	o.onFailed = append(o.onFailed, fn)
}

// OnProcessStarted registers a callback for when a process starts.
func (o *Orchestrator) OnProcessStarted(fn func(*Process)) {
	o.callbackMu.Lock()
	defer o.callbackMu.Unlock()
	o.onStarted = append(o.onStarted, fn)
}

// emitComplete notifies all complete callbacks.
func (o *Orchestrator) emitComplete(p *Process, result string) {
	agentName := ""
	if p.Agent != nil {
		agentName = p.Agent.Name
	}

	slog.Info("process completed",
		"process_id", p.ID,
		"agent", agentName,
		"result_length", len(result),
	)

	o.callbackMu.RLock()
	callbacks := make([]func(*Process, string), len(o.onComplete))
	copy(callbacks, o.onComplete)
	o.callbackMu.RUnlock()

	// Run callbacks synchronously first so they can access the process by name
	var wg sync.WaitGroup
	for _, fn := range callbacks {
		wg.Add(1)
		go func(f func(*Process, string)) {
			defer wg.Done()
			f(p, result)
		}(fn)
	}
	wg.Wait()

	// Unregister name AFTER callbacks complete
	if name := p.Name(); name != "" {
		o.Unregister(name)
	}

	// Leave all groups
	o.LeaveAllGroups(p)
}

// emitFailed notifies all failed callbacks.
func (o *Orchestrator) emitFailed(p *Process, err error) {
	agentName := ""
	if p.Agent != nil {
		agentName = p.Agent.Name
	}

	slog.Error("process failed",
		"process_id", p.ID,
		"agent", agentName,
		"error", err.Error(),
	)

	o.callbackMu.RLock()
	callbacks := make([]func(*Process, error), len(o.onFailed))
	copy(callbacks, o.onFailed)
	o.callbackMu.RUnlock()

	// Run callbacks synchronously first so they can access the process by name
	var wg sync.WaitGroup
	for _, fn := range callbacks {
		wg.Add(1)
		go func(f func(*Process, error)) {
			defer wg.Done()
			f(p, err)
		}(fn)
	}
	wg.Wait()

	// Unregister name AFTER callbacks complete
	if name := p.Name(); name != "" {
		o.Unregister(name)
	}

	// Leave all groups
	o.LeaveAllGroups(p)

	// Handle automatic restart if configured
	go o.handleAutoRestart(p, err)
}

// emitStarted notifies all started callbacks.
func (o *Orchestrator) emitStarted(p *Process) {
	o.callbackMu.RLock()
	callbacks := make([]func(*Process), len(o.onStarted))
	copy(callbacks, o.onStarted)
	o.callbackMu.RUnlock()

	for _, fn := range callbacks {
		go fn(p)
	}
}

// persistState saves process state.
func (o *Orchestrator) persistState() {
	if o.persistence == nil {
		return
	}

	o.mu.RLock()
	states := make([]ProcessState, 0, len(o.processes))
	for _, p := range o.processes {
		states = append(states, ProcessState{
			ID:          p.ID,
			AgentName:   p.Agent.Name,
			Task:        p.Task,
			WorkDir:     p.WorkDir,
			Status:      p.status,
			StartedAt:   p.StartedAt,
			Metrics:     p.metrics,
		})
	}
	o.mu.RUnlock()

	o.persistence.Save(states)
}

// recoverProcesses recovers processes from persistence.
func (o *Orchestrator) recoverProcesses() {
	if o.persistence == nil {
		return
	}

	states, err := o.persistence.Load()
	if err != nil {
		return
	}

	for _, state := range states {
		if state.Status == StatusRunning || state.Status == StatusPending {
			// Mark as needing restart
			// In a real implementation, we'd need agent definitions to respawn
		}
	}
}

// Persistence interface for saving process state.
type Persistence interface {
	Save(states []ProcessState) error
	Load() ([]ProcessState, error)
}

// ProcessState is the persisted state of a process.
type ProcessState struct {
	ID          string         `json:"id"`
	AgentName   string         `json:"agent_name"`
	Task        string         `json:"task"`
	WorkDir     string         `json:"work_dir"`
	Status      Status         `json:"status"`
	StartedAt   time.Time      `json:"started_at"`
	Metrics     ProcessMetrics `json:"metrics"`
}

// JSONPersistence saves state to a JSON file.
type JSONPersistence struct {
	path string
	mu   sync.Mutex
}

// NewJSONPersistence creates a new JSON file persistence.
func NewJSONPersistence(path string) *JSONPersistence {
	return &JSONPersistence{path: path}
}

// Save writes state to the file.
func (p *JSONPersistence) Save(states []ProcessState) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	data, err := json.MarshalIndent(states, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(p.path, data, 0644)
}

// Load reads state from the file.
func (p *JSONPersistence) Load() ([]ProcessState, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	data, err := os.ReadFile(p.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var states []ProcessState
	if err := json.Unmarshal(data, &states); err != nil {
		return nil, err
	}

	return states, nil
}

// rateLimiter implements token bucket rate limiting.
type rateLimiter struct {
	config    RateLimitConfig
	tokens    float64
	lastTime  time.Time
	mu        sync.Mutex
}

func newRateLimiter(config RateLimitConfig) *rateLimiter {
	return &rateLimiter{
		config:   config,
		tokens:   float64(config.RequestsPerMinute),
		lastTime: time.Now(),
	}
}

func (r *rateLimiter) allow() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(r.lastTime).Minutes()
	r.lastTime = now

	// Refill tokens
	r.tokens += elapsed * float64(r.config.RequestsPerMinute)
	if r.tokens > float64(r.config.RequestsPerMinute) {
		r.tokens = float64(r.config.RequestsPerMinute)
	}

	// Check if we have tokens
	if r.tokens >= 1 {
		r.tokens--
		return true
	}

	return false
}

// --- Named Process Registry ---

// Register associates a name with a process.
// Returns error if name is already taken.
// The process will be automatically unregistered when it exits.
func (o *Orchestrator) Register(name string, p *Process) error {
	if name == "" {
		return ErrInvalidInput
	}

	o.namesMu.Lock()
	defer o.namesMu.Unlock()

	if existing, ok := o.names[name]; ok && existing != p {
		return &ProcessError{ProcessID: p.ID, AgentName: p.Agent.Name, Err: ErrNameTaken}
	}

	o.names[name] = p
	p.mu.Lock()
	p.name = name
	p.mu.Unlock()

	return nil
}

// Unregister removes a name association.
func (o *Orchestrator) Unregister(name string) {
	o.namesMu.Lock()
	defer o.namesMu.Unlock()

	if p, ok := o.names[name]; ok {
		p.mu.Lock()
		p.name = ""
		p.mu.Unlock()
		delete(o.names, name)
	}
}

// GetByName returns a process by its registered name.
// Returns nil if no process is registered with that name.
func (o *Orchestrator) GetByName(name string) *Process {
	o.namesMu.RLock()
	defer o.namesMu.RUnlock()
	return o.names[name]
}

// RegisterAgent registers an agent definition for later respawning.
// This is required for automatic restart to work.
func (o *Orchestrator) RegisterAgent(agent Agent) {
	o.agentsMu.Lock()
	defer o.agentsMu.Unlock()
	o.agents[agent.Name] = agent
}

// GetAgent returns a registered agent by name.
func (o *Orchestrator) GetAgent(name string) (Agent, bool) {
	o.agentsMu.RLock()
	defer o.agentsMu.RUnlock()
	agent, ok := o.agents[name]
	return agent, ok
}

// --- Supervision Trees ---

// SupervisorStrategy determines how failures affect siblings.
type SupervisorStrategy int

const (
	// OneForOne restarts only the failed child
	OneForOne SupervisorStrategy = iota
	// OneForAll restarts all children when one fails
	OneForAll
	// RestForOne restarts the failed child and all children started after it
	RestForOne
)

// String returns the strategy name.
func (s SupervisorStrategy) String() string {
	switch s {
	case OneForOne:
		return "one_for_one"
	case OneForAll:
		return "one_for_all"
	case RestForOne:
		return "rest_for_one"
	default:
		return "unknown"
	}
}

// ChildRestart determines when a child should be restarted.
type ChildRestart int

const (
	// Permanent children are always restarted
	Permanent ChildRestart = iota
	// Transient children are restarted only on abnormal exit
	Transient
	// Temporary children are never restarted
	Temporary
)

// String returns the restart type name.
func (r ChildRestart) String() string {
	switch r {
	case Permanent:
		return "permanent"
	case Transient:
		return "transient"
	case Temporary:
		return "temporary"
	default:
		return "unknown"
	}
}

// ChildSpec defines how to start and supervise a child process.
type ChildSpec struct {
	// Name is the registered name for this child (optional)
	Name string
	// Agent is the agent definition to spawn
	Agent Agent
	// Restart determines when to restart this child
	Restart ChildRestart
	// Task is the initial task for the child
	Task string
	// SpawnOpts are additional options for spawning
	SpawnOpts []SpawnOption
}

// SupervisorSpec defines a supervision tree configuration.
type SupervisorSpec struct {
	// Strategy determines how failures affect siblings
	Strategy SupervisorStrategy
	// MaxRestarts is the maximum restarts within Window (0 = unlimited)
	MaxRestarts int
	// Window is the time window for counting restarts
	Window time.Duration
	// Children are the child specifications
	Children []ChildSpec
	// Backoff configures delay between restarts
	Backoff BackoffConfig
}

// Supervisor manages a group of child processes with automatic restart.
type Supervisor struct {
	spec         SupervisorSpec
	orchestrator *Orchestrator
	process      *Process // The supervisor's own process (optional)

	children    []*supervisedChild
	childrenMu  sync.RWMutex
	failures    []time.Time
	failuresMu  sync.Mutex
	restarts    int
	lastBackoff time.Duration

	ctx    context.Context
	cancel context.CancelFunc
}

// supervisedChild tracks a supervised process.
type supervisedChild struct {
	spec    ChildSpec
	process *Process
	index   int // Position in children slice (for RestForOne)
}

// NewSupervisor creates a new supervisor with the given spec.
func (o *Orchestrator) NewSupervisor(spec SupervisorSpec) *Supervisor {
	ctx, cancel := context.WithCancel(o.ctx)
	return &Supervisor{
		spec:         spec,
		orchestrator: o,
		children:     make([]*supervisedChild, 0, len(spec.Children)),
		ctx:          ctx,
		cancel:       cancel,
	}
}

// Start spawns all children and begins supervision.
func (s *Supervisor) Start() error {
	s.childrenMu.Lock()
	defer s.childrenMu.Unlock()

	for i, childSpec := range s.spec.Children {
		child, err := s.spawnChild(childSpec, i)
		if err != nil {
			// Shutdown already-started children
			s.stopAllChildrenLocked()
			return err
		}
		s.children = append(s.children, child)
	}

	return nil
}

// spawnChild spawns a single child and sets up monitoring.
func (s *Supervisor) spawnChild(spec ChildSpec, index int) (*supervisedChild, error) {
	// Build spawn options
	opts := make([]SpawnOption, 0, len(spec.SpawnOpts)+1)
	opts = append(opts, spec.SpawnOpts...)
	if spec.Task != "" {
		opts = append(opts, WithTask(spec.Task))
	}

	// Spawn the process
	proc, err := s.orchestrator.Spawn(spec.Agent, opts...)
	if err != nil {
		return nil, err
	}

	// Register name if specified
	if spec.Name != "" {
		if err := s.orchestrator.Register(spec.Name, proc); err != nil {
			proc.Stop()
			return nil, err
		}
	}

	child := &supervisedChild{
		spec:    spec,
		process: proc,
		index:   index,
	}

	// Set up monitoring from supervisor
	proc.SetTrapExit(false) // Children don't trap exits
	s.monitorChild(child)

	return child, nil
}

// monitorChild sets up exit monitoring for a child.
func (s *Supervisor) monitorChild(child *supervisedChild) {
	// We'll use the orchestrator's OnProcessFailed callback mechanism
	// plus direct monitoring
	go func() {
		proc := child.process

		// Wait for process to complete or fail
		for {
			select {
			case <-s.ctx.Done():
				return
			case <-time.After(DefaultSupervisorPollInterval):
				status := proc.Status()
				if status == StatusCompleted || status == StatusFailed {
					s.handleChildExit(child, status)
					return
				}
			}
		}
	}()
}

// handleChildExit is called when a supervised child exits.
func (s *Supervisor) handleChildExit(child *supervisedChild, status Status) {
	// Determine if we should restart
	shouldRestart := false
	switch child.spec.Restart {
	case Permanent:
		shouldRestart = true
	case Transient:
		shouldRestart = status == StatusFailed
	case Temporary:
		shouldRestart = false
	}

	if !shouldRestart {
		return
	}

	// Check restart limits
	if !s.canRestart() {
		// Exceeded max restarts - supervisor gives up
		s.Stop()
		return
	}

	// Calculate backoff
	backoff := s.calculateBackoff()
	if backoff > 0 {
		select {
		case <-s.ctx.Done():
			return
		case <-time.After(backoff):
		}
	}

	// Apply restart strategy
	switch s.spec.Strategy {
	case OneForOne:
		s.restartChild(child)
	case OneForAll:
		s.restartAllChildren()
	case RestForOne:
		s.restartChildAndFollowing(child)
	}
}

// canRestart checks if we're within restart limits.
func (s *Supervisor) canRestart() bool {
	if s.spec.MaxRestarts <= 0 {
		return true // Unlimited
	}

	s.failuresMu.Lock()
	defer s.failuresMu.Unlock()

	now := time.Now()
	s.failures = append(s.failures, now)

	// Prune old failures outside window
	if s.spec.Window > 0 {
		cutoff := now.Add(-s.spec.Window)
		newFailures := make([]time.Time, 0, len(s.failures))
		for _, t := range s.failures {
			if t.After(cutoff) {
				newFailures = append(newFailures, t)
			}
		}
		s.failures = newFailures
	}

	return len(s.failures) <= s.spec.MaxRestarts
}

// calculateBackoff returns the delay before next restart.
func (s *Supervisor) calculateBackoff() time.Duration {
	s.failuresMu.Lock()
	defer s.failuresMu.Unlock()

	s.restarts++

	if s.spec.Backoff.Initial == 0 {
		return 0
	}

	var delay time.Duration
	switch s.spec.Backoff.Type {
	case BackoffExponential:
		multiplier := s.spec.Backoff.Multiplier
		if multiplier == 0 {
			multiplier = 2.0
		}
		delay = time.Duration(float64(s.spec.Backoff.Initial) * pow(multiplier, float64(s.restarts-1)))
	case BackoffLinear:
		delay = s.spec.Backoff.Initial * time.Duration(s.restarts)
	case BackoffConstant:
		delay = s.spec.Backoff.Initial
	}

	if s.spec.Backoff.Max > 0 && delay > s.spec.Backoff.Max {
		delay = s.spec.Backoff.Max
	}

	s.lastBackoff = delay
	return delay
}

// pow is a simple power function for floats.
func pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}

// restartChild restarts a single child.
func (s *Supervisor) restartChild(child *supervisedChild) {
	s.childrenMu.Lock()
	defer s.childrenMu.Unlock()

	// Stop old process if still running
	if child.process.Status() == StatusRunning {
		child.process.Stop()
	}

	// Unregister old name
	if child.spec.Name != "" {
		s.orchestrator.Unregister(child.spec.Name)
	}

	// Spawn new process
	newChild, err := s.spawnChild(child.spec, child.index)
	if err != nil {
		// Failed to restart - will be handled by next failure
		return
	}

	// Update child reference
	for i, c := range s.children {
		if c == child {
			s.children[i] = newChild
			break
		}
	}
}

// restartAllChildren stops and restarts all children (OneForAll).
func (s *Supervisor) restartAllChildren() {
	s.childrenMu.Lock()
	defer s.childrenMu.Unlock()

	// Stop all children in reverse order
	for i := len(s.children) - 1; i >= 0; i-- {
		child := s.children[i]
		if child.process.Status() == StatusRunning {
			child.process.Stop()
		}
		if child.spec.Name != "" {
			s.orchestrator.Unregister(child.spec.Name)
		}
	}

	// Clear children slice
	s.children = s.children[:0]

	// Restart all children in order
	for i, childSpec := range s.spec.Children {
		newChild, err := s.spawnChild(childSpec, i)
		if err != nil {
			// Failed to restart - will be handled by next failure
			continue
		}
		s.children = append(s.children, newChild)
	}
}

// restartChildAndFollowing restarts the failed child and all after it (RestForOne).
func (s *Supervisor) restartChildAndFollowing(failed *supervisedChild) {
	s.childrenMu.Lock()
	defer s.childrenMu.Unlock()

	// Find the index of the failed child
	failedIndex := failed.index

	// Stop all children from failedIndex onwards in reverse order
	for i := len(s.children) - 1; i >= failedIndex; i-- {
		child := s.children[i]
		if child.process.Status() == StatusRunning {
			child.process.Stop()
		}
		if child.spec.Name != "" {
			s.orchestrator.Unregister(child.spec.Name)
		}
	}

	// Truncate children slice
	s.children = s.children[:failedIndex]

	// Restart from failedIndex onwards
	for i := failedIndex; i < len(s.spec.Children); i++ {
		newChild, err := s.spawnChild(s.spec.Children[i], i)
		if err != nil {
			continue
		}
		s.children = append(s.children, newChild)
	}
}

// Stop stops the supervisor and all its children.
func (s *Supervisor) Stop() {
	s.cancel()

	s.childrenMu.Lock()
	defer s.childrenMu.Unlock()

	s.stopAllChildrenLocked()
}

// stopAllChildrenLocked stops all children (must hold childrenMu).
func (s *Supervisor) stopAllChildrenLocked() {
	// Stop in reverse order
	for i := len(s.children) - 1; i >= 0; i-- {
		child := s.children[i]
		if child.process.Status() == StatusRunning {
			child.process.Stop()
		}
		if child.spec.Name != "" {
			s.orchestrator.Unregister(child.spec.Name)
		}
	}
	s.children = nil
}

// Children returns the current supervised processes.
func (s *Supervisor) Children() []*Process {
	s.childrenMu.RLock()
	defer s.childrenMu.RUnlock()

	procs := make([]*Process, len(s.children))
	for i, child := range s.children {
		procs[i] = child.process
	}
	return procs
}

// --- Dynamic Child Management ---

// ChildInfo contains information about a supervised child.
type ChildInfo struct {
	Name    string
	ID      string
	Status  Status
	Restart ChildRestart
	Agent   string
}

// WhichChildren returns information about all current children.
func (s *Supervisor) WhichChildren() []ChildInfo {
	s.childrenMu.RLock()
	defer s.childrenMu.RUnlock()

	infos := make([]ChildInfo, len(s.children))
	for i, child := range s.children {
		infos[i] = ChildInfo{
			Name:    child.spec.Name,
			ID:      child.process.ID,
			Status:  child.process.Status(),
			Restart: child.spec.Restart,
			Agent:   child.spec.Agent.Name,
		}
	}
	return infos
}

// StartChild dynamically adds and starts a new child to the supervisor.
// Returns the new process or an error if the child couldn't be started.
func (s *Supervisor) StartChild(spec ChildSpec) (*Process, error) {
	s.childrenMu.Lock()
	defer s.childrenMu.Unlock()

	// Check for duplicate name
	if spec.Name != "" {
		for _, child := range s.children {
			if child.spec.Name == spec.Name {
				return nil, &ProcessError{AgentName: spec.Agent.Name, Err: ErrNameTaken}
			}
		}
	}

	// Spawn the child
	index := len(s.children)
	child, err := s.spawnChild(spec, index)
	if err != nil {
		return nil, err
	}

	// Add to children list
	s.children = append(s.children, child)

	// Add to spec for restart purposes
	s.spec.Children = append(s.spec.Children, spec)

	return child.process, nil
}

// TerminateChild stops a specific child by name.
// The child will be restarted according to its restart policy unless DeleteChild is called.
func (s *Supervisor) TerminateChild(name string) error {
	s.childrenMu.Lock()
	defer s.childrenMu.Unlock()

	for _, child := range s.children {
		if child.spec.Name == name {
			if child.process.Status() == StatusRunning {
				child.process.Stop()
			}
			return nil
		}
	}

	return ErrProcessNotFound
}

// RestartChild forces a restart of a specific child by name.
func (s *Supervisor) RestartChild(name string) error {
	s.childrenMu.Lock()

	var targetChild *supervisedChild
	var targetIndex int
	for i, child := range s.children {
		if child.spec.Name == name {
			targetChild = child
			targetIndex = i
			break
		}
	}

	if targetChild == nil {
		s.childrenMu.Unlock()
		return ErrProcessNotFound
	}

	// Stop the current process
	if targetChild.process.Status() == StatusRunning {
		targetChild.process.Stop()
	}

	// Unregister name
	if targetChild.spec.Name != "" {
		s.orchestrator.Unregister(targetChild.spec.Name)
	}

	s.childrenMu.Unlock()

	// Spawn new process (outside lock to avoid deadlock)
	newChild, err := s.spawnChild(targetChild.spec, targetIndex)
	if err != nil {
		return err
	}

	// Update the children slice
	s.childrenMu.Lock()
	s.children[targetIndex] = newChild
	s.childrenMu.Unlock()

	return nil
}

// DeleteChild removes a child from the supervisor entirely.
// The child is stopped if running and will not be restarted.
func (s *Supervisor) DeleteChild(name string) error {
	s.childrenMu.Lock()
	defer s.childrenMu.Unlock()

	for i, child := range s.children {
		if child.spec.Name == name {
			// Stop if running
			if child.process.Status() == StatusRunning {
				child.process.Stop()
			}

			// Unregister name
			if child.spec.Name != "" {
				s.orchestrator.Unregister(child.spec.Name)
			}

			// Remove from children slice
			s.children = append(s.children[:i], s.children[i+1:]...)

			// Remove from spec
			for j, spec := range s.spec.Children {
				if spec.Name == name {
					s.spec.Children = append(s.spec.Children[:j], s.spec.Children[j+1:]...)
					break
				}
			}

			// Update indices for remaining children
			for j := i; j < len(s.children); j++ {
				s.children[j].index = j
			}

			return nil
		}
	}

	return ErrProcessNotFound
}

// CountChildren returns the number of children (total, running, failed).
func (s *Supervisor) CountChildren() (total, running, failed int) {
	s.childrenMu.RLock()
	defer s.childrenMu.RUnlock()

	total = len(s.children)
	for _, child := range s.children {
		switch child.process.Status() {
		case StatusRunning:
			running++
		case StatusFailed:
			failed++
		}
	}
	return
}

// GetChild returns the process for a specific child by name.
func (s *Supervisor) GetChild(name string) *Process {
	s.childrenMu.RLock()
	defer s.childrenMu.RUnlock()

	for _, child := range s.children {
		if child.spec.Name == name {
			return child.process
		}
	}
	return nil
}

// --- Automatic Restart Integration ---

// SpawnSupervised spawns a process with automatic restart on failure.
// The agent must be registered with RegisterAgent for restart to work.
func (o *Orchestrator) SpawnSupervised(agent Agent, restart ChildRestart, opts ...SpawnOption) (*Process, error) {
	// Register agent for respawning
	o.RegisterAgent(agent)

	// Spawn the process
	proc, err := o.Spawn(agent, opts...)
	if err != nil {
		return nil, err
	}

	// Store restart policy on process
	proc.mu.Lock()
	proc.restartPolicy = restart
	proc.spawnOpts = opts
	proc.mu.Unlock()

	return proc, nil
}

// handleAutoRestart is called when a supervised process fails.
// It should be called from emitFailed.
func (o *Orchestrator) handleAutoRestart(p *Process, err error) {
	p.mu.RLock()
	restartPolicy := p.restartPolicy
	spawnOpts := p.spawnOpts
	agentName := ""
	if p.Agent != nil {
		agentName = p.Agent.Name
	}
	procName := p.name
	p.mu.RUnlock()

	// Check if we should restart
	shouldRestart := false
	switch restartPolicy {
	case Permanent:
		shouldRestart = true
	case Transient:
		// Restart on error, not on normal completion
		shouldRestart = true
	case Temporary:
		shouldRestart = false
	default:
		// No restart policy set
		return
	}

	if !shouldRestart {
		return
	}

	// Get the agent definition
	agent, ok := o.GetAgent(agentName)
	if !ok {
		return // Can't restart without agent definition
	}

	// Check supervision policy
	if p.Supervision != nil {
		if !p.Supervision.recordFailure(p, err) {
			return // Max restarts exceeded
		}

		// Calculate and apply backoff
		backoff := p.Supervision.prepareRestart(p)
		if backoff > 0 {
			time.Sleep(backoff)
		}
	}

	// Spawn replacement
	go func() {
		newProc, spawnErr := o.Spawn(agent, spawnOpts...)
		if spawnErr != nil {
			return
		}

		// Copy restart settings to new process
		newProc.mu.Lock()
		newProc.restartPolicy = restartPolicy
		newProc.spawnOpts = spawnOpts
		newProc.Supervision = p.Supervision
		newProc.mu.Unlock()

		// Re-register name if was named
		if procName != "" {
			o.Register(procName, newProc)
		}
	}()
}

// --- Process Groups ---

// ProcessGroup enables multi-agent collaboration by grouping related processes.
// Processes can join multiple groups and groups support broadcast operations.
type ProcessGroup struct {
	name    string
	members map[string]*Process // map[processID]*Process
	mu      sync.RWMutex

	// Callbacks for membership changes
	onJoin  []func(*Process)
	onLeave []func(*Process)
}

// GroupMember contains information about a group member.
type GroupMember struct {
	ID     string
	Name   string // Registered name, if any
	Agent  string
	Status Status
}

// NewGroup creates a new process group.
// Groups are typically accessed via the orchestrator's Join/Leave methods.
func NewGroup(name string) *ProcessGroup {
	return &ProcessGroup{
		name:    name,
		members: make(map[string]*Process),
	}
}

// Name returns the group name.
func (g *ProcessGroup) Name() string {
	return g.name
}

// Join adds a process to this group.
// Returns true if the process was added, false if already a member.
func (g *ProcessGroup) Join(p *Process) bool {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, exists := g.members[p.ID]; exists {
		return false
	}

	g.members[p.ID] = p

	// Store group membership on process
	p.mu.Lock()
	if p.groups == nil {
		p.groups = make(map[string]*ProcessGroup)
	}
	p.groups[g.name] = g
	p.mu.Unlock()

	// Notify join callbacks
	for _, fn := range g.onJoin {
		go fn(p)
	}

	return true
}

// Leave removes a process from this group.
// Returns true if the process was removed, false if not a member.
func (g *ProcessGroup) Leave(p *Process) bool {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, exists := g.members[p.ID]; !exists {
		return false
	}

	delete(g.members, p.ID)

	// Remove group from process
	p.mu.Lock()
	delete(p.groups, g.name)
	p.mu.Unlock()

	// Notify leave callbacks
	for _, fn := range g.onLeave {
		go fn(p)
	}

	return true
}

// Members returns all processes in this group.
func (g *ProcessGroup) Members() []*Process {
	g.mu.RLock()
	defer g.mu.RUnlock()

	procs := make([]*Process, 0, len(g.members))
	for _, p := range g.members {
		procs = append(procs, p)
	}
	return procs
}

// MemberInfo returns information about all members.
func (g *ProcessGroup) MemberInfo() []GroupMember {
	g.mu.RLock()
	defer g.mu.RUnlock()

	infos := make([]GroupMember, 0, len(g.members))
	for _, p := range g.members {
		info := GroupMember{
			ID:     p.ID,
			Name:   p.Name(),
			Status: p.Status(),
		}
		if p.Agent != nil {
			info.Agent = p.Agent.Name
		}
		infos = append(infos, info)
	}
	return infos
}

// Count returns the number of members in the group.
func (g *ProcessGroup) Count() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.members)
}

// Has checks if a process is a member of this group.
func (g *ProcessGroup) Has(p *Process) bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	_, exists := g.members[p.ID]
	return exists
}

// OnJoin registers a callback for when processes join.
func (g *ProcessGroup) OnJoin(fn func(*Process)) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.onJoin = append(g.onJoin, fn)
}

// OnLeave registers a callback for when processes leave.
func (g *ProcessGroup) OnLeave(fn func(*Process)) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.onLeave = append(g.onLeave, fn)
}

// Broadcast sends a message to all group members.
// Returns a map of process ID to result/error.
func (g *ProcessGroup) Broadcast(ctx context.Context, message string) map[string]error {
	members := g.Members()
	results := make(map[string]error, len(members))

	var wg sync.WaitGroup
	var resultsMu sync.Mutex

	for _, p := range members {
		wg.Add(1)
		go func(proc *Process) {
			defer wg.Done()
			_, err := proc.Send(ctx, message)
			resultsMu.Lock()
			results[proc.ID] = err
			resultsMu.Unlock()
		}(p)
	}

	wg.Wait()
	return results
}

// --- Orchestrator Group Methods ---

// JoinGroup adds a process to a named group.
// Creates the group if it doesn't exist.
func (o *Orchestrator) JoinGroup(groupName string, p *Process) {
	o.groupsMu.Lock()
	group, exists := o.groups[groupName]
	if !exists {
		group = NewGroup(groupName)
		o.groups[groupName] = group
	}
	o.groupsMu.Unlock()

	group.Join(p)
}

// LeaveGroup removes a process from a named group.
func (o *Orchestrator) LeaveGroup(groupName string, p *Process) error {
	o.groupsMu.RLock()
	group, exists := o.groups[groupName]
	o.groupsMu.RUnlock()

	if !exists {
		return ErrGroupNotFound
	}

	group.Leave(p)
	return nil
}

// LeaveAllGroups removes a process from all groups.
// This is called automatically when a process exits.
func (o *Orchestrator) LeaveAllGroups(p *Process) {
	p.mu.RLock()
	groups := make([]*ProcessGroup, 0, len(p.groups))
	for _, g := range p.groups {
		groups = append(groups, g)
	}
	p.mu.RUnlock()

	for _, g := range groups {
		g.Leave(p)
	}
}

// GetGroup returns a process group by name.
func (o *Orchestrator) GetGroup(name string) (*ProcessGroup, bool) {
	o.groupsMu.RLock()
	defer o.groupsMu.RUnlock()
	group, exists := o.groups[name]
	return group, exists
}

// GetOrCreateGroup returns a group, creating it if necessary.
func (o *Orchestrator) GetOrCreateGroup(name string) *ProcessGroup {
	o.groupsMu.Lock()
	defer o.groupsMu.Unlock()

	if group, exists := o.groups[name]; exists {
		return group
	}

	group := NewGroup(name)
	o.groups[name] = group
	return group
}

// DeleteGroup removes an empty group.
// Returns error if the group has members.
func (o *Orchestrator) DeleteGroup(name string) error {
	o.groupsMu.Lock()
	defer o.groupsMu.Unlock()

	group, exists := o.groups[name]
	if !exists {
		return ErrGroupNotFound
	}

	if group.Count() > 0 {
		return &ProcessError{Err: errors.New("cannot delete non-empty group")}
	}

	delete(o.groups, name)
	return nil
}

// ListGroups returns the names of all groups.
func (o *Orchestrator) ListGroups() []string {
	o.groupsMu.RLock()
	defer o.groupsMu.RUnlock()

	names := make([]string, 0, len(o.groups))
	for name := range o.groups {
		names = append(names, name)
	}
	return names
}

// GroupMembers returns members of a named group.
func (o *Orchestrator) GroupMembers(groupName string) ([]*Process, error) {
	o.groupsMu.RLock()
	group, exists := o.groups[groupName]
	o.groupsMu.RUnlock()

	if !exists {
		return nil, ErrGroupNotFound
	}

	return group.Members(), nil
}

// BroadcastToGroup sends a message to all members of a group.
func (o *Orchestrator) BroadcastToGroup(ctx context.Context, groupName, message string) (map[string]error, error) {
	o.groupsMu.RLock()
	group, exists := o.groups[groupName]
	o.groupsMu.RUnlock()

	if !exists {
		return nil, ErrGroupNotFound
	}

	return group.Broadcast(ctx, message), nil
}
